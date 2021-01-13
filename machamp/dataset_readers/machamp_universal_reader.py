import copy
import logging
from typing import Dict, Iterable, List, Optional

from allennlp.data import DatasetReader
from allennlp.data.fields import Field, LabelField, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from transformers.models.bert.tokenization_bert import BasicTokenizer

from machamp.dataset_readers.lemma_edit import gen_lemma_rule
from machamp.dataset_readers.reader_utils import seqs2data, lines2data, mlm_mask
from machamp.dataset_readers.sequence_multilabel_field import SequenceMultiLabelField

logger = logging.getLogger(__name__)


@DatasetReader.register("machamp_universal_reader")
class MachampUniversalReader(DatasetReader):
    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 is_raw: bool = False,
                 datasets: Dict = None,
                 do_lowercase: bool = False,
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
        # TODO check use of different tokenizers
        # TODO some parts are very redundant:
        # text_to_instance2 and text_to_instance
        # read_seq2seq and read_classification
        # many things are redundant in read_* functions
        # too many seq2seq specific params
        super().__init__()

        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.is_raw = is_raw
        self.datasets = datasets
        self.do_lowercase = do_lowercase

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
        is_test = file_path == 'TESTPLACEHOLDER'
        for dataset in self.datasets:
            if is_train:
                file_path = self.datasets[dataset]['train_data_path']
            if is_dev:
                if 'validation_data_path' not in self.datasets[dataset]:
                    input_field = TextField([Token('_')], self.token_indexers)
                    sent_tasks = {'is_train': False, 'no_dev': True}
                    fields = {}
                    fields['tokens'] = input_field
                    fields['dataset'] = LabelField(dataset, label_namespace='dataset')
                    fields["metadata"] = MetadataField(sent_tasks)
                    yield Instance(fields)
                    continue
                file_path = self.datasets[dataset]['validation_data_path']
            if is_test:
                file_path = self.datasets[dataset]['test_data_path']

            num_classification = 0
            num_mlm = 0
            num_s2s = 0
            for task in self.datasets[dataset]['tasks']:
                is_clas = self.datasets[dataset]['tasks'][task]['task_type'] == 'classification'
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
                read_function = self.read_raw
            # read classification data
            elif num_tasks == num_classification:
                read_function = self.read_classification
            # read raw data for MLM
            elif num_mlm != 0:
                read_function = self.read_unlabeled
            # read seq2seq data
            elif num_s2s != 0:
                read_function = self.read_seq2seq
            # read word-level annotation (conll-like)
            else:
                read_function = self.read_sequence

            for item in read_function(dataset, file_path, is_train, max_sents):
                yield item

    def read_raw(self, dataset, path, is_train, max_sents):
        """
        Reads the data from a raw txt file. Assumes that each sentence is on a line, and
        the words are separated by a whitespace.
        """
        data = []
        word_idx = self.datasets[dataset]['word_idx']
        for sent_counter, sent in enumerate(open(path, encoding='utf-8', mode='r')):
            if max_sents != 0 and sent_counter > max_sents:
                break
            sent = sent.strip('\n')
            if self.do_lowercase:
                sent = sent.lower()
            # could also use basictokenizer?
            sent_tok = [word for word in sent.split(' ')]

            sent_tasks = {'tokens': []}
            for word in sent_tok:
                sent_tasks['tokens'].append(word)

            col_idxs = {'word_idx': word_idx}
            task2type = {}
            for task in self.datasets[dataset]['tasks']:
                task_idx = self.datasets[dataset]['tasks'][task]['column_idx']
                task_type = self.datasets[dataset]['tasks'][task]['task_type']
                task2type[task] = task_type
                col_idxs[task] = task_idx
            data.append(self.text_to_instance(sent_tasks, sent_tok, col_idxs, is_train, task2type, dataset))
        return data

    def read_classification(self, dataset, path, is_train, max_sents):
        """
        Reads classification data, meaning that it reads input text from N columns, and
        a corresponding label from a specific column.
        """
        data = []
        sent_idxs = self.datasets[dataset]['sent_idxs']
        for sent_counter, instance in enumerate(lines2data(path, self.do_lowercase)):
            task2type = {}
            if max_sents != 0 and sent_counter > max_sents:
                break

            full_text = self.tokenizer.tokenize(instance[sent_idxs[0]].strip())
            for sent_idx in sent_idxs[1:]:
                new_text = self.tokenizer.tokenize(instance[sent_idx].strip())
                full_text = self.tokenizer.add_special_tokens(full_text, new_text)
            sent_tasks = {}
            if len(full_text) == 0:
                full_text = self.tokenizer.tokenize(self.tokenizer.tokenizer.unk_token)
            sent_tasks['tokens'] = full_text

            dataset_embeds = []
            if 'dataset_embed_idx' in self.datasets[dataset]:
                if self.datasets[dataset]['dataset_embed_idx'] == -1:
                    dataset_embeds = [dataset] * len(full_text) # TODO, this *len is just to make it passable as SequenceLabelField
                else:
                    for info_piece in instance[self.datasets[dataset]['dataset_embed_idx']].split('|'):
                        if info_piece.startswith('dataset_embed='):
                            dataset_embeds = [info_piece.split('=')[1]] * len(full_text)
                            break
                if len(dataset_embeds) != len(full_text):
                    logger.error('dataset embeddings couldnt be read properly, see the documentation for more information.')
                    exit(1)


            col_idxs = {}
            for task in self.datasets[dataset]['tasks']:
                task_idx = self.datasets[dataset]['tasks'][task]['column_idx']
                task_type = self.datasets[dataset]['tasks'][task]['task_type']
                task2type[task] = task_type
                col_idxs[task] = task_idx
                if task_type == 'classification':
                    sent_tasks[task] = instance[task_idx]
                else:
                    logger.error('Task type ' + task_type + ' for task ' + task + ' in dataset ' +
                                 dataset + ' is unknown')
            data.append(self.text_to_instance(sent_tasks, instance, col_idxs, is_train, task2type, dataset, dataset_embeds))
        return data

    def read_seq2seq(self, dataset, path, is_train, max_sents):
        """
        Reads generation data. This means that both the input and the output can be a sequence
        of words. For now it only supports one input column, multiple tasks (outputs) on the
        same dataset are already supported though.
        """
        data = []
        self._source_max_exceeded = 0
        self._target_max_exceeded = 0
        logger.info("Reading instances from lines in file at: {}".format(path))
        for line_num, instance in enumerate(lines2data(path, self.do_lowercase)):
            if max_sents != 0 and line_num > max_sents:
                break

            source_string = instance[self.datasets[dataset]['sent_idxs'][0]]
            if len(self.datasets[dataset]['sent_idxs']) > 1:
                logger.error("unfortunately we do not support specifying multiple sent_idxs " +
                             "for seq2seq yet, try copying them to the same column")
            # TODO support more than 1 input? see read_classification on how
            if len(source_string) == 0:
                continue

            tokenized_source = self._source_tokenizer.tokenize(source_string.strip())
            if self._source_max_tokens and len(tokenized_source) > self._source_max_tokens:
                self._source_max_exceeded += 1
                tokenized_source = tokenized_source[: self._source_max_tokens]
            tokenized_source = self.tokenizer.add_special_tokens(tokenized_source)

            # TODO
            if 'dataset_embed_idx' in self.datasets[dataset]:
                logger.error("Sorry, the use of dataset embeddings is not implemented for the seq2seq task type yet")
                exit(1)

            sent_tasks = {'tokens': tokenized_source}
            task2type = {}
            col_idxs = {}
            for task in self.datasets[dataset]['tasks']:
                task_idx = self.datasets[dataset]['tasks'][task]['column_idx']
                task_type = self.datasets[dataset]['tasks'][task]['task_type']
                task2type[task] = task_type
                col_idxs[task] = task_idx
                if task_idx >= len(instance):
                    logger.warning("line is ignored, because it doesnt include target task: \n" + '\t'.join(instance))
                    continue
                target_string = instance[task_idx]
                if len(target_string) == 0:
                    continue
                tokenized_target = self._target_tokenizer.tokenize(target_string.strip())
                if self._target_max_tokens and len(tokenized_target) > self._target_max_tokens:
                    self._target_max_exceeded += 1
                    tokenized_target = tokenized_target[: self._target_max_tokens]
                tokenized_target = self._target_tokenizer.add_special_tokens(tokenized_target)
                sent_tasks[task] = tokenized_target
            if len(sent_tasks) > 1:
                data.append(self.text_to_instance(sent_tasks, instance, col_idxs, is_train, task2type, dataset))

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

    def read_unlabeled(self, dataset, path, is_train, max_sents):
        """
        Reads raw data to perform masked language modeling on. This is a separate function, because
        it also already masks the data.
        """
        # TODO make full use of batch size
        data = []
        sep_token = self.tokenizer.tokenize(self.tokenizer.tokenizer.sep_token)[0]
        cls_token = self.tokenizer.tokenize(self.tokenizer.tokenizer.cls_token)[0]
        mask_token = self.tokenizer.tokenize(self.tokenizer.tokenizer.mask_token)[0]

        for sent_idx, sent in enumerate(open(path, encoding='utf-8', mode='r')):
            if self.do_lowercase:
                sent = sent.lower()
            # skip empty lines
            if len(sent.strip()) == 0:
                continue
            if max_sents != 0 and sent_idx >= max_sents:
                break

            tokens = [cls_token] + self.tokenizer.tokenize(sent) + [sep_token]
            # R: 106 is taken from mbert, hope its sufficient in most cases?
            targets = mlm_mask(tokens, 106, self.tokenizer.tokenizer.vocab_size, mask_token)

            # set them to TOKENIZED, so that they are not tokenized again in the indexer/embedder
            new_tokens = []
            for token in tokens:
                new_tokens.append(Token(text=token.text, idx=token.idx, idx_end=token.idx_end,
                                        ent_type_='TOKENIZED', text_id=token.text_id, type_id=token.type_id))
            tokens = new_tokens
            if len(self.datasets[dataset]['tasks']) > 1:
                logger.error('currently MaChAmp does not support mlm mixed with other tasks on one dataset')
            task_name = list(self.datasets[dataset]['tasks'].items())[0][0]

            sent_tasks = {task_name: targets, 'tokens': tokens}
            # task2type = {task_name: 'unsupervised'}
            col_idxs = {task_name: -1}
            data.append(self.text_to_instance2(sent_tasks, col_idxs, is_train, dataset, sent))

        return data

    def text_to_instance2(self, sent_tasks, col_idxs, is_train, dataset, full_data):
        """
        This is a copy of text_to_instance, just meant for read_unsupervised().
        They should definitely be merged in the future #TODO
        """
        task_name = ''
        for item in sent_tasks:
            if item != 'tokens':
                task_name = item
        if task_name == '':
            logger.error('somehow the mlm task-name is not found, it is not allowed to be ' +
                         'called \'tokens\', please rename if this is the case')
        targets = sent_tasks[task_name]
        tokens = sent_tasks['tokens']
        input_field = TextField(tokens, self.token_indexers)
        target_field = SequenceLabelField(targets, input_field, label_namespace='mlm')

        sent_tasks["full_data"] = full_data
        sent_tasks["col_idxs"] = col_idxs
        sent_tasks['is_train'] = is_train
        sent_tasks['no_dev'] = False

        fields = {'tokens': input_field, task_name: target_field}
        fields['dataset'] = LabelField(dataset, label_namespace='dataset')
        fields["metadata"] = MetadataField(sent_tasks)

        return Instance(fields)

    def read_sequence(self, dataset, path, is_train, max_sents):
        """
        Reads conllu-like files. It relies heavily on reader_utils.seqs2data.
        Can also read sentence classification tasks for which the labels should
        be specified in the comments.
        Note that this read corresponds to a variety of task_types, but the
        differences between them during data reading are kept minimal
        """
        data = []
        word_idx = self.datasets[dataset]['word_idx']
        sent_counter = 0
        tknzr = BasicTokenizer()
            

        for sent, full_data in seqs2data(path, self.do_lowercase):
                
            task2type = {}
            sent_counter += 1
            if max_sents != 0 and sent_counter > max_sents:
                break
            sent_tasks = {}

            for token in sent:
                if len(token) <= word_idx:
                    logger.error("A sentence in the data is ill-formed:" + ' '.join(['\n'.join(x) for x in sent]))
                    exit(1)

            tokens = [token[word_idx] for token in sent]
            for tokenIdx in range(len(tokens)):
                if len(tknzr._clean_text(tokens[tokenIdx])) == 0:
                    tokens[tokenIdx] = self.tokenizer.tokenizer.unk_token
            sent_tasks['tokens'] = [Token(token) for token in tokens]

            dataset_embeds = []
            if 'dataset_embed_idx' in self.datasets[dataset]:
                for tok in sent:
                    if self.datasets[dataset]['dataset_embed_idx'] == -1:
                        dataset_embeds.append(dataset)
                    else:
                        for info_piece in tok[self.datasets[dataset]['dataset_embed_idx']].split('|'):
                            if info_piece.startswith('dataset_embed='):
                                dataset_embeds.append(info_piece.split('=')[1])
                if len(dataset_embeds) != len(tokens):
                    logger.error('dataset embeddings couldnt be read properly, see the documentation for more information.')
                    exit(1)

            col_idxs = {'word_idx': word_idx}
            for task in self.datasets[dataset]['tasks']:
                sent_tasks[task] = []
                task_type = self.datasets[dataset]['tasks'][task]['task_type']
                task_idx = self.datasets[dataset]['tasks'][task]['column_idx']
                task2type[task] = task_type
                col_idxs[task] = task_idx
                if task_type == 'classification' and task_idx == -1:
                    start = '# ' + task + ': '
                    for line in full_data:
                        if line[0].startswith(start):
                            sent_tasks[task] = line[0][len(start):]
                elif task_type in ['seq', 'multiseq', 'seq_bio']:
                    for word_data in sent:
                        if len(word_data) <= task_idx: 
                            logger.error("A sentence in the data is ill-formed:" + ' '.join(['\n'.join(x) for x in sent]))
                            exit(1)
                        sent_tasks[task].append(word_data[task_idx])
                elif task_type == 'string2string':
                    for word_data in sent:
                        if len(word_data) <= task_idx: 
                            logger.error("A sentence in the data is ill-formed:" + ' '.join(['\n'.join(x) for x in sent]))
                            exit(1)
                        task_label = gen_lemma_rule(word_data[word_idx], word_data[task_idx])
                        sent_tasks[task].append(task_label)
                elif task_type == 'dependency':
                    heads = []
                    rels = []
                    for word_data in sent:
                        if not word_data[task_idx].isdigit():
                            logger.error("Your dependency file " + path + " seems to contain invalid structures sentence "  + str(sent_counter) + " contains a non-integer head: " +   word_data[task_idx] + "\nIf you directly used UD data, this could be due to special EUD constructions which we do not support, you can clean your conllu file by using scripts/misc/cleanconl.py")
                            exit(1)
                        heads.append(int(word_data[task_idx]))
                        rels.append(word_data[task_idx + 1])
                    sent_tasks[task] = list(zip(rels, heads))
                else:
                    logger.error('Task type ' + task_type + ' for task ' + task +
                                 ' in dataset ' + dataset + ' is unknown')
            data.append(self.text_to_instance(sent_tasks, full_data, col_idxs, is_train, task2type, dataset, dataset_embeds))
        return data

    def text_to_instance(self,  # type: ignore
                         sent_tasks: Dict,
                         full_data: List[str],
                         col_idxs: Dict[str, int],
                         is_train: bool,
                         task2type: Dict[str, str],
                         dataset: str,
                         dataset_embeds: List[str] = []
                         ) -> Instance:
        """
        converts the previously read data into an AllenNLP Instance, containing mainly
        a TextField and one or more *LabelField's
        """
        fields: Dict[str, Field] = {}

        tokens = TextField(sent_tasks['tokens'], self.token_indexers)

        for task in sent_tasks:
            if task == 'tokens':
                fields[task] = tokens
                task_types = [task2type[task] for task in task2type]
                if 'seq2seq' in task_types:
                    fields['src_words'] = SequenceLabelField([str(x) for x in sent_tasks[task]],
                                                         tokens, label_namespace="src_tokens")

            elif task2type[task] == 'dependency':
                fields[task + '_rels'] = SequenceLabelField([x[0] for x in sent_tasks[task]],
                                                            tokens, label_namespace=task + '_rels')
                fields[task + '_head_indices'] = SequenceLabelField([x[1] for x in sent_tasks[task]],
                                                                    tokens, label_namespace=task + '_head_indices')

            elif task2type[task] == "multiseq":
                label_sequence = []

                # For each token label, check if it is a multilabel and handle it
                for raw_label in sent_tasks[task]:
                    label_list = raw_label.split("|")
                    label_sequence.append(label_list)
                fields[task] = SequenceMultiLabelField(label_sequence, tokens, label_namespace=task)

            elif task2type[task] == 'classification':
                fields[task] = LabelField(sent_tasks[task], label_namespace=task)

            elif task2type[task] == 'seq2seq':
                fields['target'] = TextField(sent_tasks[task], self._target_token_indexers)
                fields['target_words'] = SequenceLabelField([str(x) for x in sent_tasks[task]],
                                                                    fields['target'],
                                                                    label_namespace="target_words")

            else:  # seq labeling
                fields[task] = SequenceLabelField(sent_tasks[task], tokens, label_namespace=task)

        fields['dataset'] = LabelField(dataset, label_namespace='dataset')
        if len(dataset_embeds) != 0:
            fields['dataset_embeds'] = SequenceLabelField(dataset_embeds, tokens, label_namespace='dataset_embeds')

        sent_tasks["full_data"] = full_data
        sent_tasks["col_idxs"] = col_idxs
        sent_tasks['is_train'] = is_train
        sent_tasks['no_dev'] = False
        fields["metadata"] = MetadataField(sent_tasks)
        return Instance(fields)
