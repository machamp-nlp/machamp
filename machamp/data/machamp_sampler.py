import logging
import math
import random

from typing import Iterator, List, Tuple

from torch.utils.data import Sampler

from machamp.data.machamp_dataset import MachampDataset

logger = logging.getLogger(__name__)



class MachampBatchSampler(Sampler):
    def __init__(self,
                 datasource: MachampDataset,
                 batch_size: int,
                 max_words: int,
                 shuffle: bool,
                 smoothing_factor: float,
                 sort_by_size: bool, 
                 diverse: bool,
                 is_training: bool): 
        """
        Sampler build on MachampDatasets. Main functionality is to do 
        dataset smoothing.

        Parameters
        ----------
        datasource: MachampDataset
            A MachampDataset instance (which could hold multiple datasets)
        batch_size: int
            The number of lines (instances) to include per batch.
        max_words: int
            The maximum amount of words to have in 1 batch, this has a 
            large effect on the gpu-ram used.
        shuffle: bool
            If disable, do not shuffle the data (for predictions for example)
        smoothing_factor: float
            Value between 0.0 and 1.0 for the multinomial smoothing of the
            dataset size. 0.0 would result in equal sizes for all datasets, 
            and 1.0 in the original sizes.
        sort_by_size: bool
            Whether to sort by size. This can make training more efficient.
        diverse: bool
            Whether to have diverse batching, which means multiple 
            tasks/datasets can be included in a single batch.
        is_training: bool
            Saves whether we are training, so that we know whether to skip
            extremely long instances or not.
        """
        super().__init__()

        self.datasource = datasource
        self.batch_size = batch_size
        self.max_words = max_words
        self.shuffle = shuffle
        self.smoothing_factor = smoothing_factor
        self.sort_by_size = sort_by_size
        self.diverse = diverse 
        self.is_training = is_training

        self.dataset_sizes = {}
        dataset_orig_sizes = [(dataset, len(self.datasource.datasets[dataset])) for dataset in sorted(self.datasource.datasets)]
        total_size = sum([datasetSize[1] for datasetSize in dataset_orig_sizes])
        
        # we do not use lazy batching anymore, as it is hard to get __length__ for that
        # A batch consists of a list of tuples that hold the dataset and the instance index
        self.batches = []

        # dev samplers can be empty, then we are done here
        self.first_filled = True
        if total_size == 0:
            return

        total_new_prob = 0.0
        for dataset, size in dataset_orig_sizes:
            pi = size / total_size
            total_new_prob += math.pow(pi, self.smoothing_factor)

        for dataset, size in dataset_orig_sizes:
            pi = size / total_size
            prob = (1 / pi) * (math.pow(pi, self.smoothing_factor) / total_new_prob)
            self.dataset_sizes[dataset] = int(size * prob)

        self.fill_batches() 

    def fill_batches(self):
        self.datasource.increase_epoch()
        self.batches = []
        if self.diverse:
            self.prep_batches_diverse()
        else:
            self.prep_batches()

    def prep_batches(self):
        dataset_batches = {}
        # This will have as keys the dataset, and as values lists of batches
        for dataset in self.datasource.datasets:
            dataset_data = self.datasource.datasets[dataset]
            if self.sort_by_size:
                dataset_data.data.sort(key=lambda x: x.__len__())

            dataset_batches[dataset] = [[]]
            inst_idx = 0
            num_words_batch = 0
            indices = list(range(len(dataset_data)))
            if self.shuffle and not self.sort_by_size:
                random.shuffle(indices)
            for inst_idx in indices:
                inst_length = len(self.datasource.datasets[dataset][inst_idx])
                if self.is_training and inst_length > self.batch_size * self.max_words:
                    logger.info('skipping instance with size > batch_size*max_words: ' + 
                    str(len(self.datasource.datasets[dataset][inst_idx])) + '(' + dataset + ')')
                    inst_idx += 1
                    continue

                if len(dataset_batches[dataset][-1]) >= self.batch_size or num_words_batch + inst_length > self.max_words:
                    dataset_batches[dataset].append([])
                    num_words_batch = 0
                num_words_batch += inst_length
                dataset_batches[dataset][-1].append((dataset, inst_idx))
                inst_idx += 1
        # Don't do sampling, this happens for dev sets, but also for train
        # it could be confusing to have metrics/losses over different sets 
        # each epoch.
        if self.smoothing_factor == 1.0:
            for dataset in dataset_batches:
                for batch_idx in range(len(dataset_batches[dataset])):
                    self.batches.append(dataset_batches[dataset][batch_idx])
            # this is odd while training if shuffle is off, it gets first dataset1 then 2 then 3
            if self.shuffle:
                random.shuffle(self.batches)

        else:
            # we do not check for self.shuffle, as with smoothing we always shuffle
            for dataset in dataset_batches:
                random.shuffle(dataset_batches[dataset])
            total_size = sum(self.dataset_sizes.values())
            # We call this before the main loop with k=total size for efficiency reasons
            num_batches = sum([len(dataset_batches[dataset]) for dataset in dataset_batches])
            batches_from_datasets = random.choices(list(self.dataset_sizes.keys()), self.dataset_sizes.values(), k=num_batches)
            cur_batch = []
            len_cur_batch = 0
            # We remember the index of the instance we are at for each dataset, 
            # and we loop through them. This avoids too many redundant samples
            dataset_indices = {dataset: 0 for dataset in self.dataset_sizes}
            for cur_dataset in batches_from_datasets:
                next_batch = dataset_indices[cur_dataset]
                dataset_indices[cur_dataset]+=1
                # reset if end of dataset is reached
                if next_batch +1 >= len(dataset_batches[cur_dataset]):
                    dataset_indices[cur_dataset] = 0
    
                self.batches.append(dataset_batches[cur_dataset][next_batch])

    def prep_batches_diverse(self):
        total_size = sum(self.dataset_sizes.values())
        # We call this before the main loop with k=total size for efficiency reasons
        instances_from_datasets = random.choices(list(self.dataset_sizes.keys()), self.dataset_sizes.values(), k=total_size)
        cur_batch = []
        len_cur_batch = 0
        # We remember the index of the instance we are at for each dataset, 
        # and we loop through them. This avoids too many redundant samples
        dataset_indices = {dataset: 0 for dataset in self.dataset_sizes}
        for cur_dataset in instances_from_datasets:
            next_inst = dataset_indices[cur_dataset]
            # reset if end of dataset is reached
            if next_inst >= len(self.datasource.datasets[cur_dataset]):
                next_inst = 0
                dataset_indices[cur_dataset] = 0

            next_inst_len = len(self.datasource.datasets[cur_dataset][next_inst])

            # check if cur_batch has space:
            if len_cur_batch + next_inst_len > self.max_words or len(cur_batch) >= self.batch_size:    
                self.batches.append(cur_batch)
                cur_batch = []
                len_cur_batch = 0
                    
            dataset_indices[cur_dataset] += 1

            len_cur_batch += next_inst_len
            cur_batch.append((cur_dataset, next_inst))

        if cur_batch != []:
            self.batches.append(cur_batch)
        
    def __iter__(self) -> Iterator[List[Tuple[str, int]]]:
        """
        Iterate over the batches that are stored in self.batches.

        Returns
        -------
        batches: Iterator[List[str, int]]
            An iterator of list of tuples, the list is of length
            batch size, and each tuple consists of the dataset
            name and the index of the batch.
        """
        if self.first_filled:
            self.first_filled = False
        else:
            self.fill_batches()
        for batch in self.batches:
            yield batch

    def __len__(self) -> int:
        """
        The length of the sampler is defined as the amount of batches
        for all datasets.
        
        Returns
        -------
        num_batches
            The number of batches in all datasets.
        """
        return len(self.batches)

