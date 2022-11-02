import copy
import logging
from typing import List

import torch
from transformers import AutoTokenizer

from machamp.data.machamp_instance import MachampInstance
from machamp.data.machamp_vocabulary import MachampVocabulary

logger = logging.getLogger(__name__)


def lines2data(input_file: str, skip_first_line: bool = False):
    """
    Simply reads a tab-separated text file. Returns each line split
    by a '\t' character.

    Parameters
    ----------
    input_file: str
        The path to the file to read.
    skip_first_line: bool
        In some csv/tsv files, the heads are included in the first row.
        This option let you skip these.

    Returns
    -------
    full_data: List[str]
        A list with the columns read in the file. 
    """
    for line in open(input_file, mode='r', encoding='utf-8', errors='ignore'):
        if skip_first_line:
            skip_first_line = False
            continue
        if len(line.strip()) < 2:
            continue
        tok = [part for part in line.strip('\n').split('\t')]
        yield tok


def read_classification(
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
    Reads sentence level annotated files. We assume there is one sentence/
    utterance in each line, and annotations are separated with tabs ('\t').

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
    if max_words != -1:
        logger.error(
            "max_words defined for a classification task, this is not supported, as we do not know what the words are")
        exit(1)

    sent_idxs = config['sent_idxs']
    subword_counter = 0
    unk_counter = 0
    test_tok = tokenizer.encode_plus('a', 'b')
    has_start_token = len(tokenizer.prepare_for_model([])['input_ids']) == 2
    has_end_token = len(tokenizer.prepare_for_model([])['input_ids']) >= 1
    has_unk_token = tokenizer.unk_token != None
    has_seg_ids = 'token_type_ids' in test_tok and 1 in test_tok['token_type_ids']
    if 'skip_first_line' not in config:
        config['skip_first_line'] = False
    sent_counter = 0
    for sent_counter, data_instance in enumerate(lines2data(data_path, config['skip_first_line'])):
        if max_sents != -1 and sent_counter > max_sents:
            break
        # We use the following format 
        # input: <CLS> sent1 <SEP> sent2 <SEP> sent3 ...
        # type_ids: 0 0 .. 1 1 .. 0 0 .. 1 1 ..
        full_input = []
        seg_ids = [0]
        for counter, sent_idx in enumerate(sent_idxs):
            if sent_idx >= len(data_instance):
                logger.error(
                    'line ' + dataset + ':' + str(sent_idx) + ' doesnt\'t contain enough columns, column ' + str(
                        sent_idx) + ' is missing, should contain input.')
                exit(1)
            encoding = tokenizer.encode(data_instance[sent_idx].strip())[1:-1]
            subword_counter += len(encoding)
            if tokenizer.sep_token_id != None:
                encoding = encoding + [copy.deepcopy(tokenizer.sep_token_id)]
            if len(encoding) == 0:
                logger.warning("empty sentence found in line " + str(
                    sent_counter) + ', column ' + str(sent_idx) + ' replaced with UNK token')
                if has_unk_token:
                    encoding.append(tokenizer.unk_token_id)

            if has_seg_ids:
                seg_ids.extend([counter % 2] * len(encoding))
            full_input.extend(encoding)
        unk_counter += full_input.count(tokenizer.unk_token_id)
        if has_end_token:
            full_input = full_input[:-1]
        full_input = tokenizer.prepare_for_model(full_input)['input_ids']
        full_input = torch.tensor(full_input, dtype=torch.long)
        seg_ids = torch.tensor(seg_ids, dtype=torch.long)

        # if 'dec_dataset_embed_idx' in config and config['dec_dataset_embed_idx'] != -1:
        #    instance.add_field('dec_dataset_embeds', SequenceLabelField([data_instance[config['dec_dataset_embed_idx']]
        #                                   for _ in full_input]), input_field, label_namespace= 'dec_dataset_embeds')
        # if 'enc_dataset_embed_idx' in config and config['enc_dataset_embed_idx'] != -1:
        #    instance.add_field('enc_dataset_embeds', SequenceLabelField([data_instance[config['enc_dataset_embed_idx']]
        #                                   for _ in full_input]), input_field, label_namespace= 'dec_dataset_embeds')

        if sent_counter == 0 and is_train:
            for task in config['tasks']:  # sep. loop for efficiency
                vocabulary.create_vocab(task, True)
        golds = {}
        col_idxs = {}
        for task in config['tasks']:
            task_type = config['tasks'][task]['task_type']
            task_idx = config['tasks'][task]['column_idx']
            # if were not training we do not need annotation
            if task_idx >= len(data_instance):
                if not is_train:
                    col_idxs[task] = task_idx
                    continue
                else:
                    logger.error('Annotation for task ' + task + ' is missing in ' + dataset + ':' + str(
                        sent_counter) + ', collumn ' + str(col_idxs[task]))
                    exit(1)
            gold = data_instance[task_idx]
            if task_type == 'regression':
                try:
                    gold = float(gold)
                except ValueError:
                    logger.error('Column ' + str(col_idxs[task]) + ' in ' + dataset + ':' + str(
                        sent_idx) + " should have a float (for regression task)")
                    exit(1)
            elif task_type == 'multiclas':
                gold = torch.tensor([vocabulary.token2id(label, task, is_train) for label in gold.split('|')],
                                    dtype=torch.long)
            else:
                gold = vocabulary.token2id(gold, task, is_train)
            col_idxs[task] = task_idx
            golds[task] = gold

        data.append(MachampInstance(data_instance, full_input, seg_ids, golds, dataset))
    if is_train and max_sents != -1 and sent_counter < max_sents:
        logger.warning('Maximum sentences was set to ' + str(max_sents) + ', but dataset only contains ' + str(
            sent_counter) + ' sentences.')
    if is_train and max_words != -1 and subword_counter < max_words:
        logger.warning('Maximum (sub)words was set to ' + str(max_words) + ', but dataset only contains ' + str(
            subword_counter) + ' subwords.')

    logger.info('Stats ' + dataset + ' (' + data_path + '):')
    logger.info('Lines:    {:,}'.format(sent_counter + 1))
    logger.info('Subwords: {:,}'.format(subword_counter))
    logger.info('Unks:     {:,}'.format(unk_counter) + '\n')
    return data
