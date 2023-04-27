import datetime
import logging
from typing import Dict, Tuple, List

from torch.utils.data import Dataset
from transformers import AutoTokenizer

from machamp.data.machamp_instance import MachampInstance
from machamp.data.machamp_vocabulary import MachampVocabulary
from machamp.data.machamp_dataset import MachampDataset
from machamp.readers.read_classification import read_classification
from machamp.readers.read_mlm import read_mlm
from machamp.readers.read_sequence import read_sequence
from machamp.readers.read_raw import read_raw

logger = logging.getLogger(__name__)


# This might be neater to have in 2 classes, one DatasetCollection, and one Dataset?
class MachampDatasetCollection(Dataset):
    def __init__(self,
                 emb_name: str,
                 dataset_configs: Dict,
                 is_raw: bool = False,
                 is_train: bool = True,
                 vocabulary: MachampVocabulary = None,
                 max_input_length: int = 512,
                 raw_text: bool = False, 
                 num_epochs: int = 0):
        """
        A machamp dataset collection can hold multiple datasets. They are saved in
        self.data, which holds as keys the names of the datasets, and as values
        MachampDatasets. 

        Parameters
        ----------
        emb_name: str
            The name of the language model, so that we can get the correct tokenizer, 
            which is used in the task-specific dataset_readers
        dataset_configs: Dict
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
        num_epochs: int
            The total number of epochs we train for. This is only used for 
            the MLM task, where we train on n/num_epochs instances each epoch, 
            so we do not see the same data twice. 0 means we do not take this
            into account, and return everything (for dev/test data). We 
            increase the epoch count every time fill_batches is called.
        """

        self.tokenizer = AutoTokenizer.from_pretrained(emb_name, use_fast=False)
        self.dataset_configs = dataset_configs

        self.is_raw = is_raw
        self.is_train = is_train
        self.tasks = []
        self.task_types = []
        self.num_epochs = num_epochs

        for dataset in self.dataset_configs:
            for task in self.dataset_configs[dataset]['tasks']:
                if task in self.tasks:
                    if self.dataset_configs[dataset]['tasks'][task]['task_type'] != self.task_to_tasktype(task):
                        logger.error(
                            'Error task with same name, but different type found. Please rename and note that tasks '
                            'with the same name share the same decoder head: ' + task)
                        exit(1)
                else:
                    self.tasks.append(task)
                    self.task_types.append(dataset_configs[dataset]['tasks'][task]['task_type'])

        if vocabulary == None:
            self.vocabulary = MachampVocabulary()
        else:
            self.vocabulary = vocabulary

        self.datasets = {}
        for dataset in self.dataset_configs:
            self.datasets[dataset] = MachampDataset(dataset, self.dataset_configs[dataset], self.tokenizer, self.vocabulary, self.is_raw, self.is_train, max_input_length, self.num_epochs)


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
        The length is defined as the combined number of instances
        over all datasets. (note that the batching is handled in 
        MachampSampler)

        Returns
        -------
        length: int
            the sum of the number of instances in all datasets
        """
        return sum([len(self.datasets[dataset]) for dataset in self.datasets])

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
        return self.datasets[instance_info[0]][instance_info[1]]

    def increase_epoch(self):
        """
        Increases the epoch count for each dataset, only used in MLM tasks though, 
        because there we only see a part of the data each epoch.
        """
        for dataset in self.datasets:
            self.datasets[dataset].cur_epoch += 1

