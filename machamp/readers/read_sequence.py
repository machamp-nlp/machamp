import logging
import unicodedata
from typing import List, Dict

import torch
from transformers import AutoTokenizer
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.models.xlm_roberta.tokenization_xlm_roberta import XLMRobertaTokenizer

from machamp.utils import myutils
from machamp.data.machamp_vocabulary import MachampVocabulary
from machamp.data.machamp_instance import MachampInstance
from machamp.readers.machamp_basic_tokenizer import MachampBasicTokenizer
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

def getSpaceLocations(text):
    space_locs = []
    cur_char_idx = 0
    for char in text:
        if char == ' ':
            space_locs.append(cur_char_idx)
        else:
            cur_char_idx += 1
    return space_locs

def getSplitsFromSpaces(subwords, spaceIdxs):
    splits = []
    curCharIdx = 0
    for subword in subwords:
        new_subword = subword
        rel_pos_splits = [spaceIdx-curCharIdx for spaceIdx in spaceIdxs]
        for split_idx in reversed(rel_pos_splits):
            if split_idx < len(new_subword) and split_idx > 0:
                new_subword = new_subword[:split_idx] + ' ' + new_subword[split_idx:]
        curCharIdx += len(subword)

        #if new_subword != subword:
        splits.append([subword, new_subword])
    return splits

# TODO add documention new arguments
def get_offsets(gold_tok: List[str], subwords: List[str], norm: bool, pre_tokked):
    """
    Converts a list of gold-tokenized words and a list of subwords
    to the matching offsets as best as possible, and also produces
    gold labels for the tokenization task. The offsets are represented 
    as indices to the last subword of each word. Note that in some 
    cases there is no correct offset, and some words can refer to the 
    same subword. For example the token alot is not split in 
    XLM-r-large, and would thus have the token a and lot link with
    the same offset.
    
    Parameters
    ----------
    gold_tok: List[str]:
        The gold tokenization.
    subwords: List[str]
        The original text, but then split into subwords.
    norm: bool
        If XLM-R is used, we also have to normalize the gold, 
        otherwise it becomes impossible to match

    Returns
    -------
    offsets: List[int]
        The index of the last subword for every gold token. Should have
        the same length as annotation for sequence labeling tasks.
    tok_labels: List[bool]
        A list of labels for the tokenization task. True means: should
        merge with the following token, False for do not merge.
    """
    # TODO extract splits, and pre-split common ones
    gold_char_idx = 0
    subword_char_idx = 0
    subword_idx = 0
    offsets = []
    tok_labels = []
    if ''.join(gold_tok).replace(' ', '') != ''.join(subwords):
        
        logger.error("error, characters between original text and gold annotation are not equal")
        logger.error('orig: ' + ''.join(subwords))
        logger.error('gold: ' + ''.join(gold_tok).replace(' ', ''))
        logger.error('len(orig): ' + str(len(''.join(subwords))))
        logger.error('len(gold): ' + str(len(''.join(gold_tok).replace(' ', ''))))
        exit(1)
    # Note that this only solves 95% of all cases I guess:
    # [nononono] used to become [non ##ono ##no] and now it 
    # becomes [no no nono], which is then tokenized to [no no non ##o]
    # There are multiple ways to avoid this: pre-split more aggressively
    # by splitting subwords instead of words (1), or split all splits, instead
    # of only the non-found ones (2). Or one could find the subword ids "manually" (3)
    # all of these seem to have also unwanted effects?
    # 1) risk of splitting way too much alot -> al ot, now we learn to split al-> a l
    # 2) risk of oversplitting again, this creates a mismatch between pre-training for long tokens.
    #    this will be less than 1) I guess (at least for EN)
    # 3) This is also hardcoding too much, and be a lot of work; I guess this is the most 
    #    effective solution, but it would still not be robust when similar suffixes from 
    #    many words need to be separated (I think Polish has this)
    
    # TODO check better if this is necessary:
    if pre_tokked != None:
        spacesGold = getSpaceLocations(' '.join(gold_tok))
        spacesTok = getSpaceLocations(' '.join(subwords))
        not_found_splits = [spaceIdx for spaceIdx in spacesGold if spaceIdx not in spacesTok]
        splits = getSplitsFromSpaces(pre_tokked, not_found_splits)
    else:
        splits = {}

    if norm:
        gold_tok = [unicodedata.normalize('NFC', unicodedata.normalize('NFKD', myutils.clean_text(tok))) for tok in gold_tok]
    for word in gold_tok:
        gold_char_idx += len(word)
        while subword_char_idx < gold_char_idx:
            # links to the last subword if there is no exact match
            subword_char_idx += len(subwords[subword_idx].replace(' ', ''))
            subword_idx += 1
            if subword_char_idx < gold_char_idx:
                tok_labels.append('merge')
            else:
                tok_labels.append('split')
        offsets.append(subword_idx)
    offsets = torch.tensor(offsets, dtype=torch.long)
    return offsets, tok_labels, splits


def tok_xlmr(orig: str, pre_tokenizer: MachampBasicTokenizer, tokenizer: AutoTokenizer): 
    """
    Tokenize the original text with a XLMRobertaTokenizer, while trying to
    keep the original characters. This is only possible when the input
    is passed through utils.myutils.clean_text() first (or was already
    clean), because the XLMRobertaTokenizer does some normalization. We assume
    here that the input is already cleaned!. We replace UNKs by their 
    original string, but still return the token_id of the <unk> token.

    Parameters
    ----------
    orig: str
        The original input as a string.
    pre_tokenizer: MachampBasicTokenizer
        A tokenizer that splits punctuations. This is included even for
        XLM-R, because without it it would miss many gold tokenizations.
        For example ")." is not split in XLM-R, but in the gold tokenization
        it is. If we do not pre-split it, there is no way to get it correct
        after prediction.
    tokenizer: AutoTokenizer
        The subword tokenizer, should actually be a BertTokenizer.
    
    Returns
    -------
    no_unk_subwords: List[str]
        The subwords represented as strings. [UNK]
        tokens are not included here, but replaced by their origin.
    token_ids: List[int]
        The full list of token id's representing the input.
    """
    orig = unicodedata.normalize('NFKD', orig)
    no_unk_subwords = []
    token_ids = []
    for word in pre_tokenizer.tokenize(orig):
        tokked = tokenizer.encode(word)[1:-1]
        token_ids.extend(tokked)
        tokked = tokenizer.convert_ids_to_tokens(tokked)
        if tokked == [tokenizer.unk_token]:
            no_unk_subwords.append(word)
        else:
            for subword in tokked:
                no_unk_subwords.append(subword.replace('▁', ' '))
    return no_unk_subwords, token_ids

def tok_bert(orig: str, pre_tokenizer: MachampBasicTokenizer, tokenizer: AutoTokenizer, pre_splits):
    """
    Tokenize the original text with a BertTokenizer, while trying to
    keep the original characters. We use our own BasicTokenizer, as
    we need to skip the _clean_text call.We replace UNKs by their 
    original string, but still return the token_id of the UNK token.

    Parameters
    ----------
    orig: str
        The original input as a string.
    pre_tokenizer: MachampBasicTokenizer
        The pre-tokenizer, used so that we can identify the original 
        inputs of [UNK]s more easily (they are already separated).
    tokenizer: AutoTokenizer
        The subword tokenizer, should actually be a BertTokenizer.
    
    Returns
    -------
    no_unk_subwords: List[str]
        The subwords represented as strings. [UNK]
        tokens are not included here, but replaced by their origin.
    token_ids: List[int]
        The full list of token id's representing the input.
    """
    #orig = unicodedata.normalize('NFD', orig)
    no_unk_subwords = []
    token_ids = []

    pre_tokked = []
    for word in pre_tokenizer.tokenize(orig):
        if word in pre_splits:
            pre_tokked.extend(pre_splits[word].split(' '))
        else:
            pre_tokked.append(word)

    for word in pre_tokked:
        tokked = tokenizer.encode(word)[1:-1] 
        token_ids.extend(tokked)
        tokked = tokenizer.convert_ids_to_tokens(tokked)
        if tokked == [tokenizer.unk_token]:
            no_unk_subwords.append(word)
        else:
            for subword in tokked:
                if subword.startswith('##'):
                    no_unk_subwords.append(subword[2:])
                else:
                    no_unk_subwords.append(subword)
    return no_unk_subwords, token_ids, pre_tokked


def tokenize_and_annotate(
        full_data: List[List[str]],
        gold_tok: List[str],
        pre_tokenizer: MachampBasicTokenizer,
        tokenizer: AutoTokenizer, 
        pre_splits: Dict[str,str]):
    """
    Tokenizes the original input, and simultaneously generates annotation
    for the tokenization task. The annotation is saved as a list of True/
    False parameters for each subword. True means: merge with next, False
    means it is the end of the token. Note that the generation of the 
    offsets is non-trivial, as the splitting of subwords and subwords do
    not always overlap (see also get_offsets).

    Parameters
    ----------
    full_data: List[List[str]]
        A list with an instance for each token, which is represented as 
        a list of strings (split by '\t'). This variable includes the 
        comments in the beginning of the instance.
    gold_tok: List[str]:
        The gold tokenization.
    pre_tokenizer: MachampBasicTokenizer
        A tokenizer that splits punctuations. This is included even for
        XLM-R, because without it it would miss many gold tokenizations.
        For example ")." is not split in XLM-R, but in the gold tokenization
        it is. If we do not pre-split it, there is no way to get it correct
        after prediction.
    tokenizer: AutoTokenizer
        The tokenizer to use (that should match the used MLM).        

    Returns
    -------
    token_ids: List[str]
        The full list of token ids (for each subword, note that this can
        be longer than the annotation lists)
    token_offsets: List[str]
        The index of the last subword for every gold token. Should have
        the same length as annotation for sequence labeling tasks.
    tok_labels: List[bool]
        A list of labels for the tokenization task. True means: should
        merge with the following token, False for do not merge.
    -------
    """
    orig = ''
    for line in full_data:
        if line[0].startswith('# text =') and len(line[0]) > 9:
            orig = line[0][8:].strip()
        if line[0].startswith('# text=') and len(line[0]) > 9:
            orig = line[0][8:].strip()
    if orig == '':
        logger.error(
            'No original text found in file, altough tokenization task type is used. Make sure you have a comment in '
            'front of each instance that starts with "# text = ".')
        exit(1)

    if type(tokenizer) == XLMRobertaTokenizer:
        no_unk_subwords, token_ids, pre_tokked = tok_xlmr(orig, pre_tokenizer, tokenizer, pre_splits)
    elif type(tokenizer) == BertTokenizer:
        no_unk_subwords, token_ids, pre_tokked = tok_bert(orig, pre_tokenizer, tokenizer, pre_splits)
    else:
        logger.error("We have not implemented tokenization for the language model you choose: " + str(type(
            tokenizer)) + ". Unfortunately, this is a quite time-consuming process, because of differences in "
                          "handling special tokens/characters. You can try to use the XLMRoberta version or the Bert "
                          "version if your models tokenizer is similar, by editing the tokenize_with_gold function in "
                          "machamp/readers/read_sequence.py")
        exit(1)
    token_offsets, tok_labels, _ = get_offsets(gold_tok, no_unk_subwords, type(tokenizer)==XLMRobertaTokenizer, pre_tokked)
    return token_ids, token_offsets, tok_labels, no_unk_subwords

def get_splits(all_sents, word_col_idx, pre_tokenizer, tokenizer):
    all_splits = {}
    for sent, full_data in all_sents:
        orig_text = ''
        for line in full_data:
            if line[0].startswith('# text =') and len(line[0]) > 9:
                orig_text = line[0][8:].strip()
            if line[0].startswith('# text=') and len(line[0]) > 9:
                orig_text = line[0][8:].strip()
        gold_tok = [word[word_col_idx] for word in sent]
        no_unk_subwords, token_ids, pre_tokked = tok_bert(orig_text, pre_tokenizer, tokenizer, {})
        token_offsets, tok_labels, new_splits = get_offsets(gold_tok, no_unk_subwords, False, pre_tokked)
        for src, tgt in new_splits:
            if src not in all_splits:
                all_splits[src] = {}
            if tgt not in all_splits[src]:
                all_splits[src][tgt] = 1
            else:
                all_splits[src][tgt] += 1
    relevant_splits = {}
    # For now take the last one, but this could be tuned based on the counts? 
     # (i.e. it not be more than 10 times as uncommon as alternatives?)
    for split in all_splits:
        for tgt in all_splits[split]:
            if ' ' in tgt:
                relevant_splits[split] = tgt
    return relevant_splits

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
        pre_tokenizer = MachampBasicTokenizer(strip_accents=False, do_lower_case=False, tokenize_chinese_chars=True)
        tokenizer.do_basic_tokenize = False

    all_sents = list(seqs2data(data_path))
    if has_tok_task and is_train:
        for task in config['tasks']:
            if config['tasks'][task]['task_type'] == 'tok' and config['tasks'][task]['pre_split']:
                pre_splits = get_splits(all_sents, word_col_idx, pre_tokenizer, tokenizer)
                # TODO note that they are not per dataset as of now, might be sub-optimal for performance, 
                # but is more generalizable
                vocabulary.pre_splits.update(pre_splits)

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
            token_ids, offsets, tok_labels, no_unk_subwords = tokenize_and_annotate(full_data, [
                line[word_col_idx] for line in sent], pre_tokenizer, tokenizer, vocabulary.pre_splits)

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
