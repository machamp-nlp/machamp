import logging
from typing import Dict, Tuple, List

from torch.utils.data import Dataset
from transformers import AutoTokenizer

from machamp.data.machamp_instance import MachampInstance
from machamp.data.machamp_vocabulary import MachampVocabulary
from machamp.readers.read_classification import read_classification
from machamp.readers.read_mlm import read_mlm
from machamp.readers.read_sequence import read_sequence

logger = logging.getLogger(__name__)


class MachampDataset(Dataset):
    def __init__(self,
                 emb_name: str,
                 datasets: Dict,
                 is_raw: bool = False,
                 is_train: bool = True,
                 vocabulary: MachampVocabulary = None,
                 max_input_length: int = 512):
        """
        A machamp dataset can actually hold multiple datasets. They are saved in
        self.data, which holds as keys the names of the datasets, and as values
        a list of instances for each dataset. It relies heavily on the dataset
        readers which are defined (roughly) per task-type, they can be found
        in machamp/readers/read_*py.

        Parameters
        ----------
        emb_name: str
            The name of the language model, so that we can get the correct tokenizer, 
            which is used in the task-specific dataset_readers
        datasets: Dict[str,List[MachampInstance]]
            This is the configuration dictionary which contains all dataset
            configurations for the model.
        is_raw: bool = False
            Whether the data should be read as raw data, meaning no annotation and
            tokens simply split by whitespace.
        is_train: bool = True
            Whether we are currently training, this is important, as we have to know
            whether we have to re-use label vocabularies or create them.
        vocabulary: MachampVocabulary = None
            The vocabulary that is used for the labels of the tasks. Note that multiple
            tasks are saved in 1 vocabulary class instance.
        max_input_length: int
            The maximum input length to feed the encoder. This is only used in read_mlm, 
            as it prepares the data in the right length.
        """

        self.tokenizer = AutoTokenizer.from_pretrained(emb_name, use_fast=False)
        self.datasets = datasets

        self.is_raw = is_raw
        self.tasks = []
        self.task_types = []

        for dataset in datasets:
            for task in datasets[dataset]['tasks']:
                if task in self.tasks:
                    if datasets[dataset]['tasks'][task]['task_type'] != self.task_to_tasktype(task):
                        logger.error(
                            'Error task with same name, but different type found. Please rename and note that tasks '
                            'with the same name share the same decoder head')
                        exit(1)
                else:
                    self.tasks.append(task)
                    self.task_types.append(datasets[dataset]['tasks'][task]['task_type'])

        if vocabulary == None:
            self.vocabulary = MachampVocabulary()
        else:
            self.vocabulary = vocabulary

        self.data = {}
        for dataset in self.datasets:
            # for backwards compatibility
            if 'validation_data_path' in self.datasets[dataset]:
                self.datasets[dataset]['dev_data_path'] = self.datasets[dataset]['validation_data_path']
                del self.datasets[dataset]['validation_data_path']
            if not is_train and 'dev_data_path' not in self.datasets[dataset]:
                continue
            self.data[dataset] = []
            num_classification = 0
            num_mlm = 0
            num_s2s = 0
            for task in self.datasets[dataset]['tasks']:
                task_config = self.datasets[dataset]['tasks'][task]
                is_clas = task_config['task_type'] in ['classification', 'probdistr', 'regression', 'multiclas']
                read_seq = task_config['column_idx'] == -1 if 'column_idx' in task_config else None

                if is_clas and not read_seq:
                    num_classification += 1
                if task_config['task_type'] == 'mlm':
                    num_mlm += 1
                if task_config['task_type'] == 'seq2seq':
                    num_s2s += 1

            num_tasks = len(self.datasets[dataset]['tasks'])
            if num_mlm not in [0, num_tasks]:
                logger.error('A dataset can only consists of 0 mlm tasks or all')
            if num_s2s not in [0, num_tasks]:
                logger.error('A dataset can only consists of 0 seq2seq tasks or all')
            if num_classification not in [0, num_tasks]:
                logger.error('A dataset can only consists of 0 classification tasks or all, if you combine both ' +
                             'word-level and text-level tasks, use column_idx: -1 for the text level tasks')

            # read raw input
            # if self.is_raw:
            #    read_function = read_raw
            # read classification data
            if num_tasks == num_classification:
                read_function = read_classification
            # read raw data for MLM
            elif num_mlm != 0:
                read_function = read_mlm
            # read seq2seq data
            # elif num_s2s != 0:
            #    read_function = read_seq2seq
            # read word-level annotation (conll-like)
            else:
                read_function = read_sequence

            if is_train:
                path = self.datasets[dataset]['train_data_path']
            else:
                path = self.datasets[dataset]['dev_data_path']

            max_sents = -1 if 'max_sents' not in self.datasets[dataset] else self.datasets[dataset]['max_sents']
            max_words = -1 if 'max_words' not in self.datasets[dataset] else self.datasets[dataset]['max_words']
            for instance in read_function(dataset, self.datasets[dataset], self.tokenizer, self.vocabulary, path,
                                          is_train, max_sents, max_words, max_input_length):
                self.data[dataset].append(instance)

    def task_to_tasktype(self, task: str):
        """
        Converts a task-name (str) to its type (str)
        
        Parameters
        ----------
        task: str
            The name of the task

        Returns
        -------
        task_type: str
            The task type of the given task
        """
        task_trimmed = task.replace('-heads', '').replace('-rels', '')
        if task_trimmed in self.tasks:
            index = self.tasks.index(task_trimmed)
            return self.task_types[index]
        else:
            logger.error(task + ' not found in ' + str(self.tasks))
            exit(1)

    def __len__(self):
        """
        The length is defined as the combined number of batches
        over all datasets.

        Returns
        -------
        length: int
            the sum of the number of batches in all datasets
        """
        return sum([len(self.data[x]) for x in self.data])

    def __getitem__(self, instance_info: Tuple[str, int]):
        """
        Gets a specific instance (i.e. sentence), which is represented as an
        MachampInstance object. It can be found based on the dataset name, and its 
        index within that dataset. Because __getitem__ normally operates with
        only one argument, we used a tuple.
        
        Parameters
        ----------
        instance_info: Tuple[str, int]
            A tuple with the name of the dataset, and the index of the instance
            we are trying to get.

        Returns
        -------
        instance: MachampInstance
            a machamp instance object, containing a sentence (and its annotation).
        """
        dataset = instance_info[0]
        instance_idx = instance_info[1]
        return self.data[dataset][instance_idx]
