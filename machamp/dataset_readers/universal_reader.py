"""
A Dataset Reader for multiple dataset formats, with support for sequence labeling
tasks as well as sentence level classification.
"""

from typing import Dict, Tuple, List, Any, Callable

from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, SequenceLabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers.word_splitter import SpacyWordSplitter, WordSplitter
from allennlp.data.tokenizers import Token

from machamp.dataset_readers.lemma_edit import gen_lemma_rule
from machamp.dataset_readers.sequence_multilabel_field import SequenceMultiLabelField

import pprint
import logging

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name

"""
Reading files is done in an overly complex way, because sometimes we
have line starting with a hashtag we want to keep. We first find the
length (number of columns) of the last line of the sentence, and use 
that to decide what else is data and what is a comment. We keep both
the data and the data including the comments, so then can be written
after processing. Warning: this will break if the comment includes
the same number of tabs as the data instances.
"""

def seqs2data(conllu_file):
    sent = []
    for line in open(conllu_file):
        # because people use paste command, which includes empty tabs
        if len(line) < 2 or line.replace('\t', '') == '': 
            if len(sent) == 0:
                continue
            numCols = len(sent[-1])
            begIdx = 0
            for i in range(len(sent)):
                backIdx = len(sent) -1 -i
                if len(sent[backIdx]) == numCols:
                    begIdx = len(sent)-1-i
            yield sent[begIdx:], sent
            sent = []
        else:
            sent.append(line[:-1].split('\t'))

    # adds the last sentence when there is no empty line
    if len(sent) != 0 and sent != ['']:
        numCols = len(sent[-1])
        begIdx = 0
        for i in range(len(sent)):
            backIdx = len(sent) -1 -i
            if len(sent[backIdx]) == numCols:
                begIdx = len(sent)-1-i
        yield sent[begIdx:], sent
        sent = []

def lines2data(input_file):
    for line in open(input_file):
        tok = line.strip().split('\t')
        yield(tok)

@DatasetReader.register("machamp_universal_reader")
class UniversalDatasetReader(DatasetReader):
    def __init__(self,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False, 
                 isRaw: bool = False, 
                 tasks: Dict = None, datasets: Dict = None
                )-> None:
        super().__init__(lazy)
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self.datasets = datasets
        self.tasks = tasks
        self.isRaw = isRaw

    def read_sequence(self, dataset, path, copy_other_columns, isTrain):
        data = []
        word_idx = self.datasets[dataset]['word_idx']
        
        #ROB: TODO fix this split thing, it is super unclear
        for sent, fullData in seqs2data(path):
            sentTasks = {}
            sentTaskTypes = {}#doesnt have to be dict?
            sentTasks['words'] = []
            sentTasks['dataset'] = []
            sentTasks['isWordLevel'] = []
            for wordData in sent:
                sentTasks['dataset'].append(dataset)
                sentTasks['words'].append(wordData[word_idx])
                sentTasks['isWordLevel'].append(True)

            colIdxs = {'wordIdx': word_idx}
            for task in self.datasets[dataset]['tasks']:
                sentTasks[task] = []
                task_type = self.datasets[dataset]['tasks'][task]['task_type']
                taskIdx = self.datasets[dataset]['tasks'][task]['column_idx']
                sentTaskTypes[task] = task_type
                colIdxs[task] = taskIdx
                if task_type == 'seq':
                    for wordData in sent:
                        sentTasks[task].append(wordData[taskIdx])
                elif task_type == 'string2string':
                    for wordData in sent:
                        taskLabel = gen_lemma_rule(wordData[word_idx], wordData[taskIdx])
                        sentTasks[task].append(taskLabel)
                elif task_type == 'dependency':
                    heads = []
                    rels = []
                    for wordData in sent:
                        heads.append(wordData[taskIdx])
                        rels.append(wordData[taskIdx + 1])
                    sentTasks[task] = list(zip(rels, heads))
                elif task_type == 'multiseq':
                    for wordData in sent:
                        sentTasks[task].append(wordData[taskIdx])
                else:
                    print('Error: task type ' + task_type + ' for task ' + task + ' in dataset ' + dataset + ' is unknown')
            data.append(self.text_to_instance(sentTasks, fullData, colIdxs, sentTaskTypes, True, copy_other_columns, isTrain))
        return data

    def read_classification(self, dataset, path, copy_other_columns, isTrain):
        data = []
        sent_idxs = self.datasets[dataset]['sent_idxs']
        for instance in lines2data(path):
            full_text = []
            for sent_idx in sent_idxs:
                full_text += instance[sent_idx].split(' ') + ['[SEP]']
            full_text = full_text[:-1]

            sentTasks = {}
            sentTaskTypes = {}#doesnt have to be dict?
            sentTasks['words'] = []
            sentTasks['dataset'] = []
            sentTasks['isWordLevel'] = []
            for word in full_text:
                sentTasks['dataset'].append(dataset)
                sentTasks['words'].append(word)
                sentTasks['isWordLevel'].append(False)

            colIdxs = {}
            for task in self.datasets[dataset]['tasks']:
                sentTasks[task] = []
                taskIdx = self.datasets[dataset]['tasks'][task]['column_idx']
                task_type = self.datasets[dataset]['tasks'][task]['task_type']
                sentTaskTypes[task] = task_type
                colIdxs[task] = taskIdx
                if task_type == 'classification':
                    for word in full_text:
                        sentTasks[task].append(instance[taskIdx])
                else:
                    print('Error: task type ' + task_type + ' for task ' + task + ' in dataset ' + dataset + ' is unknown')
                    exit(1)
            data.append(self.text_to_instance(sentTasks, instance, colIdxs, sentTaskTypes, False, copy_other_columns, isTrain))
        return data
    
    def read_raw(self, dataset, path, copy_other_columns, isTrain):
        data = []
        word_idx = self.datasets[dataset]['word_idx']
        for sent in open(path):
            sentTok = sent.strip().split()
            sentTasks = {}
            sentTaskTypes = {}#doesnt have to be dict?
            sentTasks['words'] = []
            sentTasks['dataset'] = []
            sentTasks['isWordLevel'] = []
            random_task = list(self.datasets[dataset]['tasks'])[0]
            isSentLevel = self.datasets[dataset]['tasks'][random_task]['task_type'] == 'classification'
            for word in sentTok:
                sentTasks['dataset'].append(dataset)
                sentTasks['words'].append(word)
                sentTasks['isWordLevel'].append(not isSentLevel)

            colIdxs = {}
            colIdxs = {'wordIdx': word_idx}
            for task in self.datasets[dataset]['tasks']:
                taskIdx = self.datasets[dataset]['tasks'][task]['column_idx']
                task_type = self.datasets[dataset]['tasks'][task]['task_type']
                sentTaskTypes[task] = task_type
                colIdxs[task] = taskIdx
            data.append(self.text_to_instance(sentTasks, sentTok, colIdxs, sentTaskTypes, not isSentLevel, False, isTrain))
        return data

    @overrides
    def _read(self, file_path: str):
        # WARNING file_path can contain path (from predict.py) as well 
        # as split (from train.py)!

        # sentTasks is a dict with for each task a list of labels
        # entry 'dataset' contains name of dataset
        # entry 'words' contains the words
        for dataset in self.datasets:
            isTrain = file_path == 'train'
            path = file_path
            if path in self.datasets[dataset]:
                path = self.datasets[dataset][file_path]
            tasks = [self.datasets[dataset]['tasks'][x]['task_type'] for x in self.datasets[dataset]['tasks']]
            if tasks.count('classification') not in [0, len(tasks)]:
                print('ERROR, a dataset can only consists of 0 classification tasks or all')
                exit(1)
            classification = tasks.count('classification') == len(tasks)
            if self.isRaw:
                for item in self.read_raw(dataset, path, self.datasets[dataset], False):
                    yield item
            elif classification:
                for item in self.read_classification(dataset, path, self.datasets[dataset]['copy_other_columns'], isTrain):
                    yield item
            else:
                for item in self.read_sequence(dataset, path, self.datasets[dataset]['copy_other_columns'], isTrain):
                    yield item
            


    @overrides
    def text_to_instance(self,  # type: ignore
                         sentTasks: Dict[str, List[str]],
                         fullData: List[str],
                         colIdxs: Dict[str, int],
                         sentTaskTypes: Dict[str, str],
                         word_level: bool, #word level is a bit superfluous now
                         copy_other_columns: bool,
                         isTrain: bool,
                         ) -> Instance:
        fields: Dict[str, Field] = {}

        tokens = TextField([Token(w) for w in sentTasks['words']], self._token_indexers)
        words = sentTasks['words']
        #Hack, save property isWordLevel in entity field
        ent_types = [word_level] * len(words)
        tokens = TextField([Token(text=w, ent_type_=l) for w,l in zip(words, ent_types)], self._token_indexers)

        fields["tokens"] = tokens
        for task in sentTasks:
            if task in ['words', 'dataset', 'isWordLevel']:
                fields[task] = SequenceLabelField(sentTasks[task], tokens, label_namespace=task)
            elif sentTaskTypes[task] == 'dependency':
                fields['head_tags'] = SequenceLabelField([x[0] for x in sentTasks[task]],
                                                    tokens, label_namespace='head_tags')
                fields['head_indices'] = SequenceLabelField([int(x[1]) for x in sentTasks[task]], 
                                                    tokens, label_namespace='head_index_tags')
            elif sentTaskTypes[task] == "multiseq":
                label_sequence = []

                # For each token label, check if it is a multilabel and handle it
                for raw_label in sentTasks[task]:
                    label_list = raw_label.split("$")
                    label_sequence.append(label_list)
                
                fields[task] = SequenceMultiLabelField(label_sequence, tokens, label_namespace=task)
            else:
                fields[task] = SequenceLabelField(sentTasks[task], tokens, label_namespace=task)
        sentTasks["fullData"] = fullData
        sentTasks["colIdxs"] = colIdxs
        sentTasks['copy_other_columns'] = copy_other_columns
        sentTasks['isTrain'] = isTrain
        fields["metadata"] = MetadataField(sentTasks)
        #fullDataDict = {}
        #for i in range(len(fullData)):
        #    fullDataDict[i] = fullData[i]
        return Instance(fields)

