import copy
import logging
from collections import Counter
from typing import Dict, Iterable, List, Optional

from allennlp.data import DatasetReader
from allennlp.data.fields import Field, LabelField, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from transformers.models.bert.tokenization_bert import BasicTokenizer

from machamp.dataset_readers.reader_utils import lines2data
from machamp.dataset_readers.read_raw import read_raw
from machamp.dataset_readers.read_unlabeled import read_unlabeled
from machamp.dataset_readers.read_sequence import read_sequence
from machamp.dataset_readers.read_classification import read_classification
from machamp.dataset_readers.lemma_edit import gen_lemma_rule
from machamp.dataset_readers.sequence_multilabel_field import SequenceMultiLabelField

logger = logging.getLogger(__name__)


@DatasetReader.register("machamp_universal_reader")
class MachampUniversalReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 is_raw: bool = False,
                 datasets: Dict = None,
                 counting: bool = False,
                 # seq2seq task related parameters
                 target_tokenizer: Tokenizer = None,
                 target_token_indexers: Dict[str, TokenIndexer] = None,
                 source_max_tokens: Optional[int] = 128,
                 target_max_tokens: Optional[int] = 128,
                 ):
        """
        This class can be used to read a variety of formats into AllenNLP format. That is; it
        converts raw or annotated text data into Instance's which contains Fields with the input
        data as well as the gold annotation and some meta-data.
        Parameters
        ----------
        tokenizer: Tokenizer that is used; should match the one used for the indexer (only used for some data types)
        token_indexers: Used for the Fields, so that the correct indexer is used
        is_raw: When using for prediction on raw text
        datasets: The meta information for the datasets to process
        """
        super().__init__()
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.is_raw = is_raw
        self.datasets = datasets
        self.counting = counting

        # Seq2Seq task-related attributes
        self._source_tokenizer = self.tokenizer
        self._target_tokenizer = target_tokenizer or self._source_tokenizer
        self._source_token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self._target_token_indexers = target_token_indexers or self._source_token_indexers
        self._source_max_tokens = source_max_tokens
        self._target_max_tokens = target_max_tokens
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0

    def _read(self, file_path: str) -> Iterable[Instance]:
        """
        Main reading class, for each dataset it identifies the type of dataset to read,
        and calls the corresponding function. A trick is used to be able to use the same
        function universally, the actual path is read from the dataset_config based on
        which placeholder is saved in `file_path`.

        """
        is_train = file_path == 'TRAINPLACEHOLDER'
        is_dev = file_path == 'DEVPLACEHOLDER'
        for dataset in self.datasets:
            if is_train:
                file_path = self.datasets[dataset]['train_data_path']
            if is_dev:
                if 'validation_data_path' not in self.datasets[dataset]:
                    input_field = TextField([Token('_')], self.token_indexers)
                    metadata = {'is_train': False, 'no_dev': True}
                    fields = {}
                    fields['tokens'] = input_field
                    fields['dataset'] = LabelField(dataset, label_namespace='dataset')
                    fields["metadata"] = MetadataField(metadata)
                    yield Instance(fields)
                    continue
                file_path = self.datasets[dataset]['validation_data_path']

            num_classification = 0
            num_mlm = 0
            num_s2s = 0
            for task in self.datasets[dataset]['tasks']:
                is_clas = self.datasets[dataset]['tasks'][task]['task_type'] in ['classification', 'probdistr', 'regression']
                read_seq = self.datasets[dataset]['tasks'][task]['column_idx'] == -1 \
                    if 'column_idx' in self.datasets[dataset]['tasks'][task] else None
                if is_clas and not read_seq:
                    num_classification += 1
                if self.datasets[dataset]['tasks'][task]['task_type'] == 'mlm':
                    num_mlm += 1
                if self.datasets[dataset]['tasks'][task]['task_type'] == 'seq2seq':
                    num_s2s += 1

            num_tasks = len(self.datasets[dataset]['tasks'])
            if num_mlm not in [0, num_tasks]:
                logger.error('A dataset can only consists of 0 mlm tasks or all')
            if num_s2s not in [0, num_tasks]:
                logger.error('A dataset can only consists of 0 seq2seq tasks or all')
            if num_classification not in [0, num_tasks]:
                logger.error('A dataset can only consists of 0 classification tasks or all, if you combine both ' +
                             'word-level and text-level tasks, use column_idx: -1 for the text level tasks')

            max_sents = 0
            if 'max_sents' in self.datasets[dataset]:
                max_sents = self.datasets[dataset]['max_sents']

            # read raw input
            if self.is_raw:
                read_function = read_raw
            # read classification data
            elif num_tasks == num_classification:
                read_function = read_classification
            # read raw data for MLM
            elif num_mlm != 0:
                read_function = read_unlabeled
            # read seq2seq data
            elif num_s2s != 0:
                read_function = self.read_seq2seq
            # read word-level annotation (conll-like)
            else:
                read_function = read_sequence

            counting = self.counting

            # We save the number of labels to be able to do class reweighting
            # To do this, we have to loop through the instances twice, because
            # we need to know the full counts before we start.
            if counting:
                label_counter = {}
                for task in self.datasets[dataset]['tasks']:
                    label_counter[task] = Counter()
                temp_instances = []

            for instance in read_function(dataset, self.datasets[dataset], file_path, is_train, max_sents, self.token_indexers, self.tokenizer):
                instance.add_field('dataset', LabelField(dataset, label_namespace='dataset'))
                # Add dataset embeddings when index is set to -1 (then its just 
                # the dataset name
                if 'dec_dataset_embed_idx' in self.datasets[dataset] and self.datasets[dataset]['dec_dataset_embed_idx'] != -1:
                    instance.add_field('dec_dataset_embeds', SequenceLabelField([dataset for token in instance['tokens']]), instance['tokens'], label_namespace='dec_dataset_embeds')
                if 'enc_dataset_embed_idx' in self.datasets[dataset] and self.datasets[dataset]['enc_dataset_embed_idx'] != -1:
                    instance.add_field('enc_dataset_embeds', SequenceLabelField([dataset for token in instance['tokens']]), instance['tokens'], label_namespace='enc_dataset_embeds')

                if not counting:
                    yield instance
                else:
                    # count the labels
                    for task in self.datasets[dataset]['tasks']:
                        # Label counting is for all tasks except mlm, raw, and seq2seq
                        if self.datasets[dataset]['tasks'][task]["task_type"] not in ["mlm", "seq2seq", 'probdistr', 'regression']:
                            # If the task is dependency, consider _rels subtask as task for label counting
                            if self.datasets[dataset]['tasks'][task]["task_type"] == "dependency": 
                                task = task + "_rels"

                            # For task types: seq, string2string, seq_bio, dependency
                            if type(instance.fields[task]) == SequenceLabelField:
                                for label in instance.fields[task].labels:
                                    if task == "dependency_rels":
                                        label_counter["dependency"][label] += 1
                                    else:
                                        label_counter[task][label] += 1

                            # For task type: multiseq
                            elif type(instance.fields[task]) == SequenceMultiLabelField:
                                for labels in instance.fields[task].labels:
                                    for label in labels:
                                        label_counter[task][label] += 1

                            # For task type: classification
                            elif self.datasets[dataset]['tasks'][task]["task_type"] == 'classification':
                                label_counter[task][instance.fields[task].label] += 1
        
                            else:
                                logger.error("Setting class weights is only supported for sequence labeling"+ 
                                " and classification task-types, because it is unclear how it should work for" +
                                " others.")
                                exit(1)
                        

                    temp_instances.append(instance)
            if counting:
                for instance in temp_instances:
                    instance.fields["metadata"].metadata["label_counts"] = label_counter
                    yield instance


    def read_seq2seq(self, dataset, config, path, is_train, max_sents, token_indexers, tokenizer):
        """
        Reads generation data. This means that both the input and the output can be a sequence
        of words. For now it only supports one input column, multiple tasks (outputs) on the
        same dataset are already supported though.
        """
        data = []
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        logger.info("Reading instances from lines in file at: {}".format(path))
        skip_first_line = config['skip_first_line']
        for line_num, instance_input in enumerate(lines2data(path)):
            if skip_first_line:
                skip_first_line = False
                continue
            if max_sents != 0 and line_num > max_sents:
                break

            source_string = instance_input[config['sent_idxs'][0]]
            if len(config['sent_idxs']) > 1:
                logger.error("unfortunately we do not support specifying multiple sent_idxs " +
                             "for seq2seq yet, try copying them to the same column")
            # TODO support more than 1 input? see read_classification on how
            if len(source_string) == 0:
                continue

            tokenized_source = self._source_tokenizer.tokenize(source_string.strip())
            if self._source_max_tokens and len(tokenized_source) > self._source_max_tokens:
                self._source_max_exceeded += 1
                tokenized_source = tokenized_source[: self._source_max_tokens]
            tokenized_source = self._source_tokenizer.add_special_tokens(tokenized_source)
            for token in tokenized_source:
                setattr(token, 'ent_type_', 'TOKENIZED')

            input_field = TextField(tokenized_source, token_indexers)
            instance = Instance({'tokens': input_field})
            col_idxs = {}
            if len(instance_input) < 2:
                continue
            for task in config['tasks']:
                task_idx = config['tasks'][task]['column_idx']
                task_type = config['tasks'][task]['task_type']
                col_idxs[task] = task_idx
                if task_idx >= len(instance_input):
                    logger.warning("line is ignored, because it doesnt include target task: \n" + '\t'.join(instance_input))
                    continue
                target_string = instance_input[task_idx]
                if len(target_string) == 0:
                    continue
                tokenized_target = self._target_tokenizer.tokenize(target_string.strip())
                if self._target_max_tokens and len(tokenized_target) > self._target_max_tokens:
                    self._target_max_exceeded += 1
                    tokenized_target = tokenized_target[: self._target_max_tokens]
                tokenized_target = self._target_tokenizer.add_special_tokens(tokenized_target)
                for token in tokenized_target:
                    token.ent_type_='TOKENIZED'
                targetField = TextField(tokenized_target, self._target_token_indexers)
                targetSeqField = SequenceLabelField([str(x) for x in tokenized_target], targetField, label_namespace='target_words')
                instance.add_field('target', targetField)
                instance.add_field('target_words', targetSeqField)

            metadata = {}
            # the other tokens field will often only be available as word-ids, so we save a copy
            metadata['tokens'] = tokenized_source
            metadata["full_data"] = instance_input
            metadata["col_idxs"] = col_idxs
            metadata['is_train'] = is_train
            metadata['no_dev'] = False
            instance.add_field('metadata', MetadataField(metadata))

            data.append(instance)

        if self._source_max_tokens and self._source_max_exceeded:
            logger.info(
                "In {0} instances, the source token length exceeded the max limit ({1}) and were truncated.".format(
                    self._source_max_exceeded,
                    self._source_max_tokens,
                ))
        if self._target_max_tokens and self._target_max_exceeded:
            logger.info(
                "In {0} instances, the target token length exceeded the max limit ({1}) and were truncated.".format(
                    self._target_max_exceeded,
                    self._target_max_tokens,
                ))
        return data

