import logging
from typing import List, Dict

import torch
from transformers import AutoTokenizer
from transformers.models.bert.tokenization_bert import BasicTokenizer
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.xlm_roberta.tokenization_xlm_roberta import XLMRobertaTokenizer

from machamp.utils import myutils
from machamp.utils import tok_utils
from machamp.data.machamp_vocabulary import MachampVocabulary
from machamp.data.machamp_instance import MachampInstance
from machamp.utils.lemma_edit import gen_lemma_rule

logger = logging.getLogger(__name__)


def seqs2data(tabular_file: str, skip_first_line: bool = False):
    """
    Reads a conll-like file. We do not base the comment identification on
    the starting character being a '#' , as in some of the datasets we used
    the words where in column 0, and could start with a `#'. Instead we start
    at the back, and see how many columns (tabs) the file has. Then we judge
    any sentences at the start which do not have this amount of columns (tabs)
    as comments. Returns both the read column data as well as the full data.

    Parameters
    ----------
    tabular_file: str
        The path to the file to read.
    skip_first_line: bool
        In some csv/tsv files, the heads are included in the first row.
        This option let you skip these.

    Returns
    -------
    full_data: List[List[str]]
        A list with an instance for each token, which is represented as 
        a list of strings (split by '\t'). This variable includes the 
        comments in the beginning of the instance.
    instance_str: List[List[str]]
        A list with an instance for each token, which is represented as 
        a list of strings (split by '\t'). This variable does not include
        the comments in the beginning of the instance.
    """
    sent = []
    for line in open(tabular_file, mode="r", encoding="utf-8"):
        if skip_first_line:
            skip_first_line = False
            continue
        # because people use paste command, which includes empty tabs
        if len(line) < 2 or line.replace('\t', '') == '':
            if len(sent) == 0:
                continue
            num_cols = len(sent[-1])
            beg_idx = 0
            for i in range(len(sent)):
                back_idx = len(sent) - 1 - i
                if len(sent[back_idx]) == num_cols:
                    beg_idx = len(sent) - 1 - i
            yield sent[beg_idx:], sent
            sent = []
        else:
            sent.append([token for token in line.rstrip("\n").split('\t')])

    # adds the last sentence when there is no empty line
    if len(sent) != 0 and sent != ['']:
        num_cols = len(sent[-1])
        beg_idx = 0
        for i in range(len(sent)):
            back_idx = len(sent) - 1 - i
            if len(sent[back_idx]) == num_cols:
                beg_idx = len(sent) - 1 - i
        yield sent[beg_idx:], sent


def tokenize_simple(tokenizer: AutoTokenizer, sent: List[List[str]], word_col_idx: int, num_special_tokens: int, has_unk: bool):
    """
    A tokenizer that tokenizes each token separately (over gold tokenization). 
    We found that this is the most robust method to tokenize overall (handling
    of special characters, whitespaces etc.).

    Parameters
    ----------
    tokenizer: AutoTokenizer
        The tokenizer to use (that should match the used MLM).
    sent: List[List[str]]:
        Contains all information of the tokens (also annotation), hence a list
        of lists.
    word_col_idx: int:
        The column index that contains the input words.
    num_special_toks: int
        Number of special tokens, here assumed to be 2 (start/end token) or 1
        (only end token)
    has_unk: bool
        Does the tokenizer have an unk token
    
    Returns
    -------
    token_ids: List[int]
        The full list of token ids (for each subword, note that this can
        be longer than the annotation lists)
    offsets: list[int]
        The index of the last subword for every gold token. Should have
        the same length as annotation for sequence labeling tasks.
    """
    token_ids = []
    offsets = []
    for token_idx in range(len(sent)):
        # we do not use return_tensors='pt' because we do not know the length beforehand
        if num_special_tokens == 2:
            tokked = tokenizer.encode(sent[token_idx][word_col_idx])[1:-1]
        elif num_special_tokens == 1:
            # We assume that if there is only one special token, it is the end token
            tokked = tokenizer.encode(sent[token_idx][word_col_idx])[:-1]
        elif num_special_tokens == 0:
            tokked = tokenizer.encode(sent[token_idx][word_col_idx])
        else:
            logger.error('Number of special tokens is currently not handled: ' + str(num_special_tokens))
            exit(1)
        if len(tokked) == 0 and has_unk:
            tokked = [tokenizer.unk_token_id]
        token_ids.extend(tokked)
        offsets.append(len(token_ids)-1)  
    offsets = torch.tensor(offsets, dtype=torch.long)

    return token_ids, offsets


def read_sequence(
        dataset: str,
        config: dict,
        tokenizer: AutoTokenizer,
        vocabulary: MachampVocabulary,
        data_path: str,
        is_train: bool,
        max_sents: int,
        max_words: int,
        max_input_length: int):
    """
    Reads conllu-like files. It relies heavily on seqs2data for the reading
    logic.  Can also read sentence classification tasks for which the labels 
    should be specified in the comments.
    
    Parameters
    ----------
    dataset: str
        The (unique) name of the dataset.
    config: dict
        The dataset configuration, with all defined parameters we need to 
        read the file properly.
    tokenizer: AutoTokenizer
        The tokenizer to use (that should match the used MLM).
    vocabulary: MachampVocabulary
        The vocabularies for all tasks.
    data_path: str
        The path to read the data from
    is_train: bool
        Whether we are currrently training, important to know so that we can
        expand the label vocabulary or not.
    max_sents: int
        The maximum number of sentences to read.
    max_words: int
        The maximum amount of words to read, rounds down.
    max_input_length
        The maximum amount of subwords to input to the encoder, not used here.

    Returns
    -------
    data: List[Machamp.data.MachampInstance]
        A list of instances, including tokenization, annotation etc.
    """
    data = []
    word_col_idx = config['word_idx']
    sent_counter = 0
    word_counter = 0
    unk_counter = 0
    subword_counter = 0
    has_unk = tokenizer.unk_token_id != None
    has_tok_task = 'tok' in [config['tasks'][task]['task_type'] for task in config['tasks']]
    num_special_tokens = len(tokenizer.prepare_for_model([])['input_ids'])
    if has_tok_task:
        pre_tokenizer = BasicTokenizer(strip_accents=False, do_lower_case=False, tokenize_chinese_chars=True)
        tokenizer.do_basic_tokenize = False

    all_sents = list(seqs2data(data_path))
    learn_splits = False
    if has_tok_task and is_train:
        for task in config['tasks']:
            learn_splits = config['tasks'][task]['task_type'] == 'tok' and config['tasks'][task]['pre_split']

    for sent, full_data in all_sents:
        # sent is a list of lists, of shape sentenceLength, numColumns
        if max_sents != -1 and sent_counter >= max_sents and is_train:
            break
        sent_counter += 1

        for token in sent:
            if len(token) <= word_col_idx:
                logger.error("Sentence (" + str(sent_counter) + ") does not have input words in column " + str(
                    word_col_idx) + ':\n ' + ' '.join(['\n'.join(x) for x in sent]))
                exit(1)

        if has_tok_task:
            token_ids, offsets, tok_labels, no_unk_subwords, new_splits = tok_utils.tokenize_and_annotate(full_data, [
                line[word_col_idx] for line in sent], pre_tokenizer, tokenizer, vocabulary.pre_splits, learn_splits)
            # Note that the splits are not per dataset as of now, might be sub-optimal for performance, 
            # but is more generalizable. 
            # They are also not picked in a smart way; we just keep the last for each..
            if new_splits != {}:
                vocabular.pre_splits = new_splits

        else:
            token_ids, offsets = tokenize_simple(tokenizer, sent, word_col_idx, num_special_tokens, has_unk)
            no_unk_subwords = None
        token_ids = tokenizer.prepare_for_model(token_ids, return_tensors='pt')['input_ids']


        # if index = -1, the dataset name is used, and this is handled in the superclass
        # dec_dataset_embeds = []
        # if 'dec_dataset_embed_idx' in config and config['dec_dataset_embed_idx'] != -1:
        #    instance.add_field('dec_dataset_embeds', SequenceLabelField([token[config['dec_dataset_embed_idx']] for token in sent]), input_field, label_namespace='dec_dataset_embeds')
        # enc_dataset_embeds = []
        # if 'enc_dataset_embed_idx' in config and config['enc_dataset_embed_idx'] != -1:
        #    instance.add_field('enc_dataset_embeds', SequenceLabelField([token[config['enc_dataset_embed_idx']] for token in sent]), input_field, label_namespace='enc_dataset_embeds')

        col_idxs = {}
        golds = {}
        for task in config['tasks']:
            if is_train:
                vocabulary.create_vocab(task, True)
            task_type = config['tasks'][task]['task_type']
            task_idx = config['tasks'][task]['column_idx']
            col_idxs[task] = task_idx

            # Read sequence labeling tasks
            if task_type in ['seq', 'seq_bio', 'multiseq', 'string2string']:
                for word_data in sent:
                    if len(word_data) <= task_idx:
                        logger.error("Sentence (" + str(
                            sent_counter) + ") does not have annotation for task " + task + ' column ' + str(
                            task_idx) + ' is missing:\n' + ' '.join(['\n'.join(x) for x in sent]))
                        exit(1)
                    if len(word_data[task_idx]) == 0:
                        logger.error("Sentence (" + str(
                            sent_counter) + ") does not have annotation for task " + task + ' column ' + str(
                            task_idx) + ' is empty\n' + ' '.join(['\n'.join(x) for x in sent]))
                        exit(1)

                # adapt labels for string2string
                if task_type == 'string2string':
                    golds[task] = torch.tensor([vocabulary.token2id(
                        gen_lemma_rule(token_info[word_col_idx], token_info[task_idx]), task, is_train) for token_info
                                                in sent], dtype=torch.long)


                # Special handling for multiseq, as it required a different labelfield
                elif task_type == 'multiseq':
                    label_sequence = []
                    for token_info in sent:
                        label_list = token_info[task_idx].split("|")
                        label_sequence.append([vocabulary.token2id(label, task, is_train) for label in label_list])
                    max_labels = max([len(label) for label in label_sequence])
                    padded_label_sequence = [labels + [vocabulary.UNK_ID] * (max_labels-len(labels)) for labels in label_sequence]
                    golds[task] = torch.tensor(padded_label_sequence, dtype=torch.long)
                else:
                    golds[task] = torch.tensor(
                        [vocabulary.token2id(token_info[task_idx], task, is_train) for token_info in sent],
                        dtype=torch.long)

            elif task_type == 'dependency':
                heads = []
                for word_data in sent:
                    if not word_data[task_idx].isdigit():
                        logger.error(
                            "Your dependency file " + data_path + " seems to contain invalid structures sentence " +
                                str(sent_counter) + " contains a non-integer head: " + word_data[
                                task_idx] + "\nIf you directly used UD data, this could be due to multiword tokens, "
                                            "which we currently do not support, you can clean your conllu file by "
                                            "using scripts/misc/cleanconl.py")
                        exit(1)
                try:
                    heads = [int(token_info[task_idx]) for token_info in sent]
                except ValueError:
                    logger.error(
                        'Head of dependency task in sentence ' + str(sent_counter) + ' is not an integer.\n' + ' '.join(
                            ['\n'.join(x) for x in sent]))
                    exit(1)
                golds[task + '-heads'] = torch.tensor(heads, dtype=torch.long)

                if len(word_data) <= task_idx + 1:
                    logger.error("Sentence (" + str(
                        sent_counter) + ") does not have annotation for task " + task + ' column ' + str(
                        task_idx + 1) + ' is missing:\n' + ' '.join(['\n'.join(x) for x in sent]))
                    exit(1)
                elif len(word_data[task_idx + 1]) == 0:
                    logger.error("Sentence (" + str(
                        sent_counter) + ") does not have annotation for task " + task + ' column ' + str(
                        task_idx + 1) + ' is empty\n' + ' '.join(['\n'.join(x) for x in sent]))
                else:
                    golds[task + '-rels'] = torch.tensor(
                        [vocabulary.token2id(token_info[task_idx + 1], task, is_train) for token_info in sent],
                        dtype=torch.long)

            # Read sentence classification task in the comments
            elif task_type == 'classification' and task_idx == -1:
                start = '# ' + task + ': '
                label = ''
                for line in full_data:
                    if line[0].startswith(start):
                        label = line[0][len(start):]
                if label != '':
                    golds[task] = vocabulary.token2id(label, task, is_train)
                else:
                    logger.error(
                        "Classification label " + task + "not found. Make sure that every sentence has a comment "
                                                         "looking like:\n# " + task + ": <LABEL>\n")
                    exit(1)

            elif task_type == 'tok':
                golds[task] = torch.tensor(
                    [vocabulary.token2id(subword_annotation, task, is_train) for subword_annotation in tok_labels],
                    dtype=torch.long)
            else:
                logger.error('Task type ' + task_type + ' for task ' + task +
                             ' in dataset ' + dataset + ' is unknown')

        # In some (rare) cases, we have  a problem with the tokenization of the raw text, it can never
        # match the gold. For example nononono is tokenized to [CLS]', 'non', '##ono', '##non', '##ono', '[SEP]'
        # but it is tokenized to 6 tokens in EN_EWT. This means that we can never learn the right tokenization.
        # But even worse, we can't learn the other tasks on top of the tokenization of the raw text.
        # So we fall back to tokenize the gold-tokenized input, so that we do not throw away data for the 
        # other (non-tokenization) tasks.
        no_mapping = False
        for task in golds:
            if len(token_ids) - num_special_tokens < len(golds[task]):
                no_mapping = True
        if no_mapping and is_train:
            # No mapping can be found, but we still want to train for the other tasks, so backoff to the gold
            # tokenization
            token_ids, offsets = tokenize_simple(tokenizer, sent, word_col_idx, num_special_tokens, has_unk)
            token_ids = tokenizer.prepare_for_model(token_ids, return_tensors='pt')['input_ids']
            no_unk_subwords =  tokenizer.convert_ids_to_tokens(token_ids)
            if type(tokenizer) == BertTokenizer:
                no_unk_subwords = [subword[:2] if subword.startswith('##') else subword for subword in no_unk_subwords] 
            elif type(tokenizer) == XLMRobertaTokenizer:
                no_unk_subwords = [subword.replace('▁', ' ') for subword in no_unk_subwords]

            if has_tok_task: # this should always be true though
                new_tok_labels = []
                for i in range(len(offsets)):
                    if i in offsets:
                        new_tok_labels.append('split')
                    else:
                        new_tok_labels.append('merge')
                for task in config['tasks']:
                    if config['tasks'][task]['task_type'] == 'tok':
                        tok_name = task
                        break
                golds[task] = torch.tensor(
                    [vocabulary.token2id(subword_annotation, task, is_train) for subword_annotation in new_tok_labels],
                    dtype=torch.long)

        if has_unk:
            unk_counter += sum(token_ids == tokenizer.unk_token_id)
        subword_counter += len(token_ids) - 2
        word_counter += len(offsets)
        if max_words != -1 and word_counter > max_words and is_train:
            break

        data.append(
            MachampInstance(full_data, token_ids, torch.zeros((len(token_ids)), dtype=torch.long), golds, dataset,
                            offsets, no_unk_subwords))

    if is_train and max_sents != -1 and sent_counter < max_sents:
        logger.warning('Maximum sentences was set to ' + str(max_sents) + ', but dataset only contains ' + str(
            sent_counter) + ' lines.')
    if is_train and max_words != -1 and word_counter < max_words:
        logger.warning('Maximum words was set to ' + str(max_words) + ', but dataset only contains ' + str(
            word_counter) + ' words.')

    logger.info('Stats ' + dataset + '(' + data_path + '):')
    logger.info('Lines:      {:,}'.format(sent_counter))
    logger.info('Words:      {:,}'.format(word_counter))
    logger.info('Subwords:   {:,}'.format(subword_counter))
    logger.info('Unks:       {:,}'.format(unk_counter))
    logger.info('Pre-splits: {:,}'.format(len(vocabulary.pre_splits)) + '\n')
    return data
