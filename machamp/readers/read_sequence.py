import logging
from typing import List

import torch
from transformers import AutoTokenizer
from transformers.models.bert.tokenization_bert import BasicTokenizer
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.xlm_roberta.tokenization_xlm_roberta import XLMRobertaTokenizer

from machamp.data.machamp_instance import MachampInstance
from machamp.data.machamp_vocabulary import MachampVocabulary
from machamp.utils import myutils
from machamp.utils import tok_utils
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
        if len(line) < 2 or line.replace('\t', '') in ['' '\n']:
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
            if line.startswith('# text'):  # because tab in UD_Munduruku-TuDeT
                line = line.replace('\t', ' ')
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


def tokenize_simple(tokenizer: AutoTokenizer, sent: List[List[str]], word_col_idx: int, num_special_tokens: int,
                    has_unk: bool):
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
    num_special_tokens: int
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
        offsets.append(len(token_ids) - 1)
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
        script_finder = tok_utils.ScriptFinder()
        type_tokenizer = myutils.identify_tokenizer(tokenizer)

    all_sents = list(seqs2data(data_path))
    do_splits = False
    if has_tok_task:
        for task in config['tasks']:
            if config['tasks'][task]['task_type'] == 'tok':
                do_splits = config['tasks'][task]['pre_split']

    
    if is_train:
        vocabulary.create_vocab('dataset_embeds', True)

    learn_splits = do_splits and is_train
    for sent, full_data in all_sents:
        # sent is a list of lists, of shape sentenceLength, numColumns
        if max_sents not in  [-1, None] and sent_counter >= max_sents and is_train:
            break
        sent_counter += 1

        for token in sent:
            if len(token) <= word_col_idx:
                logger.error("Sentence (" + str(sent_counter) + ") does not have input words in column " + str(
                    word_col_idx) + ':\n ' + ' '.join(['\n'.join(x) for x in sent]) + '\t' + data_path)
                exit(1)

        if has_tok_task:
            gold_tokens = [line[word_col_idx] for line in sent]
            token_ids, offsets, tok_labels, no_unk_subwords, new_splits = tok_utils.tokenize_and_annotate(full_data,
                                                                                                          gold_tokens,
                                                                                                          pre_tokenizer,
                                                                                                          tokenizer,
                                                                                                          vocabulary.pre_splits,
                                                                                                          learn_splits,
                                                                                                          script_finder,
                                                                                                          do_splits,
                                                                                                          type_tokenizer)
            # Note that the splits are not per dataset as of now, might be sub-optimal for performance, 
            # but is more generalizable. 
            # They are also not picked in a smart way; we just keep the last for each..
            if new_splits != {}:
                vocabulary.pre_splits = new_splits
            # We assume that if we have only one special token, that it is the end token

        else:
            token_ids, offsets = tokenize_simple(tokenizer, sent, word_col_idx, num_special_tokens, has_unk)
            no_unk_subwords = None
        token_ids = tokenizer.prepare_for_model(token_ids, return_tensors='pt')['input_ids']

        dataset_ids_subwords = []
        if 'dataset_embed_idx' in config:
            if config['dataset_embed_idx'] == -1:
                dataset_ids_words = [vocabulary.token2id(dataset, 'dataset_embeds', is_train) for token in sent]
            else:
                dataset_ids_words = [vocabulary.token2id(token[config['dataset_embed_idx']], 'dataset_embeds', is_train)
                                                                     for token in sent]
            dataset_ids_subwords = torch.zeros(len(token_ids), dtype=torch.long)

        for word_idx in range(len(offsets)):
            if word_idx == 0:
                beg = 0
            else:
                beg = offsets[word_idx-1]
            end = offsets[word_idx]
            for subword_idx in range(beg, end+1):# end+1 because inclusive
                # 1+ for the CLS token
                if 'dataset_embed_idx' in config:
                    if num_special_tokens == 2:
                        dataset_ids_subwords[1+subword_idx] = dataset_ids_words[word_idx]
                    else:
                        dataset_ids_subwords[1+subword_idx] = dataset_ids_words[word_idx]

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
                            task_idx) + ' is missing:\n' + ' '.join(['\n'.join(x) for x in sent]) + '\t' + data_path)
                        exit(1)
                    if len(word_data[task_idx]) == 0:
                        logger.error("Sentence (" + str(
                            sent_counter) + ") does not have annotation for task " + task + ' column ' + str(
                            task_idx) + ' is empty\n' + ' '.join(['\n'.join(x) for x in sent]) + '\t' + data_path)
                        exit(1)

                # adapt labels for string2string
                if task_type == 'string2string':
                    golds[task] = torch.tensor([vocabulary.token2id(
                        gen_lemma_rule(token_info[word_col_idx], token_info[task_idx]), task, is_train, warning=False) for token_info
                        in sent], dtype=torch.long)

                # Special handling for multiseq, as it required a different labelfield
                elif task_type == 'multiseq':
                    label_sequence = []
                    for token_info in sent:
                        label_list = token_info[task_idx].split("|")
                        label_sequence.append([vocabulary.token2id(label, task, is_train) for label in label_list])
                    max_labels = max([len(label) for label in label_sequence])
                    padded_label_sequence = [labels + [vocabulary.UNK_ID] * (max_labels - len(labels)) for labels in
                                             label_sequence]
                    golds[task] = torch.tensor(padded_label_sequence, dtype=torch.long)
                else:
                    golds[task] = torch.tensor(
                        [vocabulary.token2id(token_info[task_idx], task, is_train) for token_info in sent],
                        dtype=torch.long)

            elif task_type == 'dependency':
                heads = []
                try:
                    for wordIdx, word in enumerate(sent):
                        if word[task_idx] == '_':
                            head = 0 if wordIdx == 0 else 1
                        else:
                            head = int(word[task_idx])
                        heads.append(head)
                except ValueError:
                    logger.error(
                            "Your dependency file " + data_path + " seems to contain invalid structures sentence " +
                                            str(sent_counter) + " contains a non-integer head: " + word_data[
                                            task_idx] + "\nIf you directly used UD data, this could be due to " +
                                            "multiword tokens, which we currently do not support, you can clean your " +
                                            "conllu file by using scripts/misc/cleanconl.py")
                    exit(1)
                golds[task + '-heads'] = torch.tensor(heads, dtype=torch.long)

                if len(sent[0]) <= task_idx + 1:
                    logger.error("Sentence (" + str(
                        sent_counter) + ") does not have annotation for task " + task + ' column ' + str(
                        task_idx + 1) + ' is missing:\n' + ' '.join(['\n'.join(x) for x in sent]))
                    exit(1)
                elif len(sent[0][task_idx + 1]) == 0:
                    logger.error("Sentence (" + str(
                        sent_counter) + ") does not have annotation for task " + task + ' column ' + str(
                        task_idx + 1) + ' is empty\n' + ' '.join(['\n'.join(x) for x in sent]))
                else:
                    golds[task + '-rels'] = torch.tensor(
                        [vocabulary.token2id(token_info[task_idx + 1], task, is_train) for token_info in sent],
                        dtype=torch.long)

            # Read sentence classification task in the comments
            elif task_type == 'classification' and task_idx == -1:
                start1 = '# ' + task + ': '
                start2 = '# ' + task + ' = '
                label = ''
                for line in full_data:
                    if line[0].startswith(start1):
                        label = line[0][len(start1):]
                    elif line[0].startswith(start2):
                        label = line[0][len(start2):]
                if label != '':
                    golds[task] = vocabulary.token2id(label, task, is_train)
                else:
                    if is_train:
                        logger.error(
                            "Classification label " + task + "not found. Make sure that every sentence has a comment "
                                                             "looking like:\n# " + task + ": <LABEL>\n")
                        exit(1)
                    else:
                        golds[task] = vocabulary.token2id('_', task, is_train)

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
        for task in config['tasks']:
            task_type = config['tasks'][task]['task_type']
            if task_type != 'classification':
                gold_name = task if task_type != 'dependency' else task + '-heads'
                if len(token_ids) - num_special_tokens < len(golds[gold_name]):
                    no_mapping = True
        if no_mapping and is_train:
            # No mapping can be found, but we still want to train for the other tasks, so backoff to the gold
            # tokenization
            token_ids, offsets = tokenize_simple(tokenizer, sent, word_col_idx, num_special_tokens, has_unk)
            token_ids = tokenizer.prepare_for_model(token_ids, return_tensors='pt')['input_ids']
            no_unk_subwords = tokenizer.convert_ids_to_tokens(token_ids)
            if type(tokenizer) == BertTokenizer:
                no_unk_subwords = [subword[:2] if subword.startswith('##') else subword for subword in no_unk_subwords]
            elif type(tokenizer) == XLMRobertaTokenizer:
                no_unk_subwords = [subword.replace('▁', ' ') for subword in no_unk_subwords]

            if has_tok_task:  # this should always be true though
                new_tok_labels = []
                for i in range(len(offsets)):
                    if i in offsets:
                        new_tok_labels.append('split')
                    else:
                        new_tok_labels.append('merge')
                for tok_task in config['tasks']:
                    if config['tasks'][tok_task]['task_type'] == 'tok':
                        break
                golds[tok_task] = torch.tensor(
                    [vocabulary.token2id(subword_annotation, tok_task, is_train) for subword_annotation in
                     new_tok_labels],
                    dtype=torch.long)

        if has_unk:
            unk_counter += sum(token_ids == tokenizer.unk_token_id)
        subword_counter += len(token_ids) - num_special_tokens
        word_counter += len(offsets)
        if max_words != -1 and word_counter > max_words and is_train:
            break

        #if len(offsets) > 1000:
        #    print(' skipping instance of len>1000 ' + data_path)
        #    continue
        data.append(
            MachampInstance(full_data, token_ids, torch.zeros((len(token_ids)), dtype=torch.long), golds, dataset,
                            offsets, no_unk_subwords, dataset_ids_subwords))

    if is_train and max_sents != -1 and sent_counter < max_sents:
        logger.warning('Maximum sentences was set to ' + str(max_sents) + ', but dataset only contains ' + str(
            sent_counter) + ' lines.')
    if is_train and max_words != -1 and word_counter < max_words:
        logger.warning('Maximum words was set to ' + str(max_words) + ', but dataset only contains ' + str(
            word_counter) + ' words.')

    logger.info('Stats ' + dataset + ' (' + data_path + '):')
    logger.info('Lines:      {:,}'.format(sent_counter))
    logger.info('Words:      {:,}'.format(word_counter))
    logger.info('Subwords:   {:,}'.format(subword_counter))
    logger.info('Unks:       {:,}'.format(unk_counter))
    logger.info('Pre-splits: {:,}'.format(len(vocabulary.pre_splits)))
    return data
