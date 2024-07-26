import logging
from typing import List

import torch
from transformers import AutoTokenizer
from transformers.models.bert.tokenization_bert import BasicTokenizer

from machamp.data.machamp_instance import MachampInstance
from machamp.data.machamp_vocabulary import MachampVocabulary
from machamp.utils import tok_utils
from machamp.utils import myutils

logger = logging.getLogger(__name__)


def read_raw(
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
    Reads raw text files and prepares them to use as input for MaChAmp.
    
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
        The maximum amount of subwords to input to the encoder

    Returns
    -------
    data: List[Machamp.data.MachampInstance]
        A list of instances
    """
    data = []
    sent_counter = 0
    unk_counter = 0
    subword_counter = 0
    has_unk = tokenizer.unk_token != None
    num_special_tokens = len(tokenizer.prepare_for_model([])['input_ids'])
    type_tokenizer = myutils.identify_tokenizer(tokenizer)
    script_finder = tok_utils.ScriptFinder()
    pre_tokenizer = BasicTokenizer(strip_accents=False, do_lower_case=False, tokenize_chinese_chars=True)

    if is_train:
        logger.error("can't train with --raw_text, if you want to do language modeling, see the task type mlm")
        exit(1)

    for line_idx, line in enumerate(open(data_path, encoding='utf-8', errors='ignore')):
        if max_sents not in [None, -1] and line_idx >= max_sents:
            break
        line = line.strip()

        no_unk_subwords, token_ids, _ = tok_utils.tok(line, pre_tokenizer, tokenizer, {}, script_finder, False, type_tokenizer)

        offsets = torch.tensor(range(len(token_ids)), dtype=torch.long)
        # if running out of mem:
        #token_ids = token_ids[:2000]
        #no_unk_subwords = no_unk_subwords[:2000]
        token_ids = tokenizer.prepare_for_model(token_ids)['input_ids']
        token_ids = torch.tensor(token_ids, dtype=torch.long)
    
        # skip empty lines
        if len(token_ids) <= num_special_tokens:
            continue
        sent_counter += 1
        if has_unk:
            unk_counter += sum(token_ids == tokenizer.unk_token_id)
        subword_counter += len(token_ids) - num_special_tokens

        golds = {}
        full_data = []
        for token in token_ids:
            full_data.append([str(len(full_data)+1), token] + ['_'] * 8) # TODO hardcoded, doesnt work for non-UD (including classification)

        data.append(MachampInstance(full_data, token_ids, torch.zeros((len(token_ids)), dtype=torch.long), golds, dataset,
                                    offsets, no_unk_subwords))
    logger.info('Stats ' + dataset + ' (' + data_path + '):')
    logger.info('Lines:    {:,}'.format(sent_counter))
    logger.info('Subwords: {:,}'.format(subword_counter))
    logger.info('Unks:     {:,}'.format(unk_counter))
    return data
