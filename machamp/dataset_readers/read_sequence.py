import logging
from typing import Dict, List
from allennlp.data.instance import Instance
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField, LabelField
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer

from machamp.dataset_readers.lemma_edit import gen_lemma_rule
from machamp.dataset_readers.reader_utils import _clean_text
from machamp.dataset_readers.sequence_multilabel_field import SequenceMultiLabelField

logger = logging.getLogger(__name__)

def seqs2data(conllu_file, skip_first_line=False):
    """
    Reads a conllu-like file. We do not base the comment identification on
    the starting character being a '#' , as in some of the datasets we used
    the words where in column 0, and could start with a `#'. Instead we start
    at the back, and see how many columns (tabs) the file has. Then we judge
    any sentences at the start which do not have this amount of columns (tabs)
    as comments. Returns both the read column data as well as the full data.
    """
    sent = []
    for line in open(conllu_file, mode="r", encoding="utf-8"):
        if skip_first_line:
            skip_first_line=False
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

def read_sequence(dataset, config, path, is_train, max_sents, token_indexers, tokenizer):
    """
    Reads conllu-like files. It relies heavily on seqs2data for the reading
    logic.  Can also read sentence classification tasks for which the labels 
    should be specified in the comments.
    """
    data = []
    word_idx = config['word_idx']
    sent_counter = 0

    for sent, full_data in seqs2data(path, config['skip_first_line']):
        # sent is a list of lists, of shape sentenceLength, numColumns
        sent_counter += 1
        if max_sents != 0 and sent_counter > max_sents:
            break

        for token in sent:
            if len(token) <= word_idx:
                logger.error("A sentence (" + str(sent_counter) + ") in the data is ill-formed:" + ' '.join(['\n'.join(x) for x in sent]))
                exit(1)

        tokens = [token[word_idx] for token in sent]
        col_idxs = {'word_idx': word_idx}
        input_field = TextField([Token(token) for token in tokens] , token_indexers)
        instance = Instance({'tokens': input_field})

        # if index = -1, the dataset name is used, and this is handled in the superclass
        dec_dataset_embeds = []
        if 'dec_dataset_embed_idx' in config and config['dec_dataset_embed_idx'] != -1:
            instance.add_field('dec_dataset_embeds', SequenceLabelField([token[config['dec_dataset_embed_idx']] for token in sent]), input_field, label_namespace='dec_dataset_embeds')
        enc_dataset_embeds = []
        if 'enc_dataset_embed_idx' in config and config['enc_dataset_embed_idx'] != -1:
            instance.add_field('enc_dataset_embeds', SequenceLabelField([token[config['enc_dataset_embed_idx']] for token in sent]), input_field, label_namespace='enc_dataset_embeds')
        
        for task in config['tasks']:
            task_label_seq = []
            task_type = config['tasks'][task]['task_type']
            task_idx = config['tasks'][task]['column_idx']
            col_idxs[task] = task_idx
            
            # Read sequence labeling tasks
            if task_type in ['seq', 'seq_bio', 'multiseq', 'string2string']:
                labels = []
                for word_data in sent:
                    if len(word_data) <= task_idx: 
                        logger.error("A sentence (" + str(sent_counter) + ") in the data is ill-formed:" + ' '.join(['\n'.join(x) for x in sent]))
                        exit(1)
                    labels.append(word_data[task_idx])

                # adapt labels for string2string
                if task_type == 'string2string':
                    labels = [gen_lemma_rule(word, lemma) for word, lemma in zip(tokens, labels)]

                # Special handling for multiseq, as it required a different labelfield
                if task_type == 'multiseq':
                    label_sequence = []
                    # For each token label, check if it is a multilabel and handle it
                    for raw_label in labels:
                        label_list = raw_label.split("|")
                        label_sequence.append(label_list)
                    instance.add_field(task, SequenceMultiLabelField(label_sequence, input_field, label_namespace=task))
                else:
                    instance.add_field(task, SequenceLabelField(labels, input_field, label_namespace=task))

            elif task_type == 'dependency':
                heads = []
                rels = []
                for word_data in sent:
                    if not word_data[task_idx].isdigit():
                        logger.error("Your dependency file " + path + " seems to contain invalid structures sentence "  + str(sent_counter) + " contains a non-integer head: " +   word_data[task_idx] + "\nIf you directly used UD data, this could be due to multiword tokens, which we currently do not support, you can clean your conllu file by using scripts/misc/cleanconl.py")
                        exit(1)
                    heads.append(int(word_data[task_idx]))
                    rels.append(word_data[task_idx + 1])
                instance.add_field(task + '_rels', SequenceLabelField(rels, input_field, label_namespace=task + '_rels'))
                instance.add_field(task + '_head_indices', SequenceLabelField(heads, input_field, label_namespace=task + '_head_indices'))

            # Read sentence classification task in the comments
            elif task_type == 'classification' and task_idx == -1:
                start = '# ' + task + ': '
                label = ''
                for line in full_data:
                    if line[0].startswith(start):
                        label = line[0][len(start):]
                if label != '':
                    instance.add_field(task, LabelField(label, label_namespace=task))
                else:
                    logger.error("Classification label " + task + " not found. Make sure that every sentence has a comment looking like:\n# " + task + ": <LABEL>\n")
                    exit(1)
        
            else:
                logger.error('Task type ' + task_type + ' for task ' + task +
                             ' in dataset ' + dataset + ' is unknown')
        
        metadata = {}
        # the other tokens field will often only be available as word-ids, so we save a copy
        metadata['tokens'] = tokens
        metadata["full_data"] = full_data
        metadata["col_idxs"] = col_idxs
        metadata['is_train'] = is_train
        metadata['no_dev'] = False
        instance.add_field('metadata', MetadataField(metadata))

        data.append(instance)
    return data


