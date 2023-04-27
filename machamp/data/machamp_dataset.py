import datetime
import logging
from typing import Dict, Tuple, List

from torch.utils.data import Dataset
from transformers import AutoTokenizer

from machamp.data.machamp_instance import MachampInstance
from machamp.data.machamp_vocabulary import MachampVocabulary
from machamp.readers.read_classification import read_classification
from machamp.readers.read_mlm import read_mlm
from machamp.readers.read_sequence import read_sequence
from machamp.readers.read_raw import read_raw

logger = logging.getLogger(__name__)


class MachampDataset(Dataset):
    def __init__(self,
                 dataset_name: str,
                 dataset_config: Dict,
                 tokenizer: AutoTokenizer, 
                 vocabulary: MachampVocabulary = None,
                 is_raw: bool = False, 
                 is_train: bool = True,
                 max_input_length: int = 512,
                 num_epochs: int = 0):
        """
        A machamp dataset is a single dataset, which can contain multiple
        tasks.  The main data is saved in self.data, which is a list of
        MachampInstance's.  This class implements that for the mlm task type, different
        data is seen each epoch, so the number of instances per epoch == the dataset
        size/num_epochs
        

        Parameters
        ----------
        dataset_name: str
            The (unique) name of the dataset.
        dataset_config: Dict
            This is the configuration dictionary which contains all dataset
            configurations for the model.
        tokenizer: AutoTokenizer
            Tokenizer to use to read the data.
        vocabulary: MachampVocabulary = None
            The vocabulary that is used for the labels of the tasks. Note that multiple
            tasks are saved in 1 vocabulary class instance.
        is_train: bool
            If not training and no dev split available skip.
        is_raw: bool = False
            Whether the data should be read as raw data, meaning no annotation and
            tokens simply split by whitespace.
        is_train: bool = True
            Whether we are currently training, this is important, as we have to know
            whether we have to re-use label vocabularies or create them.
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
        self.dataset_config = dataset_config
        self.num_epochs = num_epochs
        self.cur_epoch = -1 # we increase this before batching 
        self.is_train = is_train
        self.is_raw = is_raw
        self.data = []
        self.is_mlm = False

        # for backwards compatibility
        if 'validation_data_path' in self.dataset_config:
            self.dataset_config['dev_data_path'] = self.dataset_config['validation_data_path']
            del self.dataset_config['validation_data_path']
        if not is_train and 'dev_data_path' not in self.dataset_config:
            return
        num_classification = 0
        num_mlm = 0
        num_s2s = 0
        for task in self.dataset_config['tasks']:
            task_config = self.dataset_config['tasks'][task]
            is_clas = task_config['task_type'] in ['classification', 'probdistr', 'regression', 'multiclas']
            read_seq = task_config['column_idx'] == -1 if 'column_idx' in task_config else None

            if is_clas and not read_seq:
                num_classification += 1
            if task_config['task_type'] == 'mlm':
                num_mlm += 1
            if task_config['task_type'] == 'seq2seq':
                num_s2s += 1

        self.is_mlm = num_mlm == 1
        num_tasks = len(self.dataset_config['tasks'])
        if num_mlm not in [0, num_tasks]:
            logger.error('A dataset can only consists of 0 mlm tasks or all')
        if num_s2s not in [0, num_tasks]:
            logger.error('A dataset can only consists of 0 seq2seq tasks or all')
        if num_classification not in [0, num_tasks]:
            logger.error('A dataset can only consists of 0 classification tasks or all, if you combine both ' +
                             'word-level and text-level tasks, use column_idx: -1 for the text level tasks')

        # read raw input
        if self.is_raw:
            read_function = read_raw
        # read classification data
        elif num_tasks == num_classification:
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
            path = self.dataset_config['train_data_path']
        else:
            path = self.dataset_config['dev_data_path']

        max_sents = -1 if 'max_sents' not in self.dataset_config else self.dataset_config['max_sents']
        max_words = -1 if 'max_words' not in self.dataset_config else self.dataset_config['max_words']

        logger.info("Reading " + path + '...')
        start_time = datetime.datetime.now()
        for instance in read_function(dataset_name, dataset_config, tokenizer, vocabulary, path,
                                        is_train, max_sents, max_words, max_input_length):
            self.data.append(instance)
        seconds = str(datetime.datetime.now() - start_time).split('.')[0]
        logger.info("Done reading " + path + " ({:.1f}s)".format((datetime.datetime.now() - start_time).seconds) + '\n')

    def __len__(self):
        """
        Length of the data is straightforward, its the number of instances
        (sentences), except for the MLM task, where we divide it by the number 
        of epochs.

        Returns
        -------
        Number of instances (sentences) in this dataset
        """
        if self.is_mlm and self.num_epochs != 0:
            return int(len(self.data)/self.num_epochs)
        else:
            return len(self.data)

    def __getitem__(self, index):
        """
        Returns the item with a specific index. For the MLM task this is a bit
        more complex, as we have to divide the data into the number of epochs. 

        Parameters
        ----------
        index: int
            The index of the instance we are looking for.

        Returns
        -------
        instance: MachampInstance
            The instance and all its data.
        """
        # note that this never shuffles over the epochs, the data is split in
        # 20 parts which are handled sequentially. Shuffling can happen within
        # these parts though (in the sampler)
        if self.is_mlm and self.num_epochs!= 0:
            return self.data[int(len(self.data)/self.num_epochs) * self.cur_epoch + index]
        else:
            return self.data[index]


