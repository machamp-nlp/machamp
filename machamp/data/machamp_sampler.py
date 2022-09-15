import math
import random
from typing import Iterator, List, Tuple

from torch.utils.data import Sampler

from machamp.data.machamp_dataset import MachampDataset


class MachampBatchSampler(Sampler):
    def __init__(self,
                 data_source: MachampDataset,
                 batch_size: int,
                 max_words: int,
                 shuffle: bool,
                 smoothing_factor: float,
                 sort_by_size: bool):
        """
        Sampler build on MachampDatasets. Main functionality is to do 
        dataset smoothing.

        Parameters
        ----------
        data_source: MachampDataset
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
        """
        super().__init__(data_source)

        self.data_source = data_source
        self.batch_size = batch_size
        self.max_words = max_words # TODO use this!
        self.batches = {}
        self.shuffle = shuffle
        self.smoothing_factor = smoothing_factor
        self.sort_by_size = sort_by_size

        for dataset in self.data_source.data:
            dataset_data = self.data_source.data[dataset]
            if sort_by_size:
                dataset_data.sort(key=lambda x: x.__len__())

            self.batches[dataset] = [[]]
            inst_idx = 0
            num_words_batch = 0
            while inst_idx < len(dataset_data):
                num_words_batch += len(self.data_source.data[dataset][inst_idx])
                if len(self.batches[dataset][-1]) > 0 and (len(self.batches[dataset][-1]) >= batch_size or num_words_batch > max_words):
                    self.batches[dataset].append([])
                    num_words_batch = 0
                self.batches[dataset][-1].append((dataset, inst_idx))
                inst_idx += 1
            if self.shuffle:
                random.shuffle(self.batches[dataset])

    def __iter__(self) -> Iterator[List[Tuple[str, int]]]:
        """
        Iterate over the batches that are stored in self.batches.
        It keeps a list of indices, with a batch index for each 
        dataset, and calculates the new_sizes based on smoothing.
        Note that this might skip the last (couple of) batch(es) 
        of a dataset in an epoch... (even when smoothing == 1.0)

        Returns
        -------
        batches: Iterator[List[str, int]]
            An iterator of list of tuples, the list is of length
            batch size, and each tuple consists of the dataset
            name and the index of the batch.
        """
        datasetSizes = [(dataset, len(self.batches[dataset])) for dataset in self.batches]
        new_sizes = []
        total_size = sum([datasetSize[1] for datasetSize in datasetSizes])
        total_new_prob = 0.0
        for dataset, size in datasetSizes:
            pi = size / total_size
            total_new_prob += math.pow(pi, self.smoothing_factor)

        for dataset, size in datasetSizes:
            pi = size / total_size
            prob = (1 / pi) * (math.pow(pi, self.smoothing_factor) / total_new_prob)
            new_sizes.append(int(size * prob))

        # Keep an index of which batch has already been used for each dataset
        dataset_batch_idxs = [0] * len(self.batches)
        for i in range(total_size):
            batch_id = random.randrange(sum(new_sizes))
            # batch_id is a random batch idx from 0 to the length of all batches
            counter = 0
            # counter is used to keep track of which dataset we need to be in
            # for example: dataset1.newsize=100, dataset2.newsize=900, if counter
            # < 100 we have to get a batch from dataset1
            for dataset, size in zip([x[0] for x in datasetSizes], new_sizes):
                new_counter = counter + size
                if counter <= batch_id < new_counter:
                    dataset_idx = [x[0] == dataset for x in datasetSizes].index(True)
                    yield self.batches[dataset][dataset_batch_idxs[dataset_idx]]
                    dataset_batch_idxs[dataset_idx] += 1
                    # if we handled all items in this dataset, restart
                    if dataset_batch_idxs[dataset_idx] >= len(self.batches[dataset]):
                        dataset_batch_idxs[dataset_idx] = 0
                    break
                counter = new_counter

    def __len__(self) -> int:
        """
        The length of the sampler is defined as the amount of batches
        for all datasets.
        
        Returns
        -------
        num_batches
            The number of batches in all datasets.
        """
        return sum([len(self.batches[x]) for x in self.batches])
