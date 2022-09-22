import logging
from typing import List

import torch
from transformers import AutoTokenizer
from transformers import DataCollatorForLanguageModeling  # or DataCollatorForWholeWordMask

from machamp.data.machamp_instance import MachampInstance
from machamp.data.machamp_vocabulary import MachampVocabulary

logger = logging.getLogger(__name__)


def read_mlm(
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
    Reads text files, and masks them with the BERT strategy, 15% of tokens are masked, 80% 
    with the mask token, 10% with a random token, and 10% it is left as is. Assumes input
    to be in text format, with an utterance per line. Note that this takes into account the 
    max length defined in the parameters configuration. For now it truncates lines, so information
    might be lost.
    
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
        A list of instances, including tokenization, annotation etc.
    """
    data = []
    sent_counter = 0
    unk_counter = 0
    subword_counter = 0
    has_unk = tokenizer.unk_token != None
    masker = DataCollatorForLanguageModeling(tokenizer)

    if len(config['tasks']) > 1:
        logger.error("MLM is currently only supported as single task on a dataset.")
        exit(1)
    task = list(config['tasks'])[0]

    for line in open(data_path):
        line = line.rstrip('\n')
        if max_sents != -1 and sent_counter >= max_sents and is_train:
            break

        token_ids = tokenizer.encode(line, return_tensors='pt')[0]

        # truncate too long sentences
        if len(token_ids) >= max_input_length:
            token_ids = token_ids[list(range(max_input_length-1)) + [len(token_ids) - 1]]

        # skip empty lines
        if len(token_ids) <= 2:
            continue
        sent_counter += 1

        if has_unk:
            unk_counter += sum(token_ids == tokenizer.unk_token_id)
        subword_counter += len(token_ids) - 2

        # if index = -1, the dataset name is used, and this is handled in the superclass
        # dec_dataset_embeds = []
        # if 'dec_dataset_embed_idx' in config and config['dec_dataset_embed_idx'] != -1:
        #    instance.add_field('dec_dataset_embeds', SequenceLabelField([token[config['dec_dataset_embed_idx']] for token in sent]), input_field, label_namespace='dec_dataset_embeds')
        # enc_dataset_embeds = []
        # if 'enc_dataset_embed_idx' in config and config['enc_dataset_embed_idx'] != -1:
        #    instance.add_field('enc_dataset_embeds', SequenceLabelField([token[config['enc_dataset_embed_idx']] for token in sent]), input_field, label_namespace='enc_dataset_embeds')

        input_text, output_labels = masker.torch_mask_tokens(token_ids.view(1, -1))
        input_text = input_text[0]
        output_labels = output_labels[0]
        task_type = config['tasks'][task]['task_type']
        golds = {task: output_labels}
        offsets = torch.arange(0,len(input_text))

        data.append(MachampInstance([line], input_text, torch.zeros((len(token_ids)), dtype=torch.long), golds, dataset, offsets))
    if is_train and max_sents != -1 and sent_counter < max_sents:
        logger.warning('Maximum sentences was set to ' + str(max_sents) + ', but dataset only contains ' + str(
            sent_counter) + ' lines. Note that this could be because empty lines are ignored')
    if is_train and max_words != -1 and subword_counter < max_words:
        logger.warning('Maximum (sub)words was set to ' + str(max_words) + ', but dataset only contains ' + str(
            subword_counter) + ' subwords.')

    logger.info('Stats ' + dataset + '(' + data_path + '):')
    logger.info('Lines:    {:,}'.format(sent_counter))
    logger.info('Subwords: {:,}'.format(subword_counter))
    logger.info('Unks:     {:,}'.format(unk_counter) + '\n')
    return data
