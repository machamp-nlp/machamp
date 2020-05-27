import logging
import random
from collections import deque
from typing import List, Tuple, Iterable, cast, Dict, Deque

from overrides import overrides
from collections import defaultdict

from allennlp.common.checks import ConfigurationError
from allennlp.common.util import add_noise_to_dict_values, lazy_groups_of
from allennlp.data.dataset import Batch
from allennlp.data.instance import Instance
from allennlp.data.iterators.data_iterator import DataIterator
from allennlp.data.vocabulary import Vocabulary

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


def group_by_dataset(instances: List[Instance]) -> List[Instance]:
    d = defaultdict(list)
    
    for item in instances:
        d[item['dataset'][0]].append(item)
    
    return [v for v in d.values()]



def sort_by_padding(instances: List[Instance],
                    sorting_keys: List[Tuple[str, str]],  # pylint: disable=invalid-sequence-index
                    vocab: Vocabulary,
                    padding_noise: float = 0.0) -> List[Instance]:
    """
    Sorts the instances by their padding lengths, using the keys in
    ``sorting_keys`` (in the order in which they are provided).  ``sorting_keys`` is a list of
    ``(field_name, padding_key)`` tuples.
    """
    instances_with_lengths = []
    for instance in instances:
        # Make sure instance is indexed before calling .get_padding
        instance.index_fields(vocab)
        padding_lengths = cast(Dict[str, Dict[str, float]], instance.get_padding_lengths())
        if padding_noise > 0.0:
            noisy_lengths = {}
            for field_name, field_lengths in padding_lengths.items():
                noisy_lengths[field_name] = add_noise_to_dict_values(field_lengths, padding_noise)
            padding_lengths = noisy_lengths
        instance_with_lengths = ([padding_lengths[field_name][padding_key]
                                  for (field_name, padding_key) in sorting_keys],
                                 instance)
        instances_with_lengths.append(instance_with_lengths)
    instances_with_lengths.sort(key=lambda x: x[0])
    return [instance_with_lengths[-1] for instance_with_lengths in instances_with_lengths]


@DataIterator.register("data-type-bucket")
class BucketIterator(DataIterator):
    """
    An iterator which by default, pads batches with respect to the maximum input lengths `per
    batch`. Additionally, you can provide a list of field names and padding keys which the dataset
    will be sorted by before doing this batching, causing inputs with similar length to be batched
    together, making computation more efficient (as less time is wasted on padded elements of the
    batch).

    Parameters
    ----------
    sorting_keys : List[Tuple[str, str]]
        To bucket inputs into batches, we want to group the instances by padding length, so that we
        minimize the amount of padding necessary per batch. In order to do this, we need to know
        which fields need what type of padding, and in what order.

        For example, ``[("sentence1", "num_tokens"), ("sentence2", "num_tokens"), ("sentence1",
        "num_token_characters")]`` would sort a dataset first by the "num_tokens" of the
        "sentence1" field, then by the "num_tokens" of the "sentence2" field, and finally by the
        "num_token_characters" of the "sentence1" field.  TODO(mattg): we should have some
        documentation somewhere that gives the standard padding keys used by different fields.
    padding_noise : float, optional (default=.1)
        When sorting by padding length, we add a bit of noise to the lengths, so that the sorting
        isn't deterministic.  This parameter determines how much noise we add, as a percentage of
        the actual padding value for each instance.
    biggest_batch_first : bool, optional (default=False)
        This is largely for testing, to see how large of a batch you can safely use with your GPU.
        This will let you try out the largest batch that you have in the data `first`, so that if
        you're going to run out of memory, you know it early, instead of waiting through the whole
        epoch to find out at the end that you're going to crash.

        Note that if you specify ``max_instances_in_memory``, the first batch will only be the
        biggest from among the first "max instances in memory" instances.
    batch_size : int, optional, (default = 32)
        The size of each batch of instances yielded when calling the iterator.
    instances_per_epoch : int, optional, (default = None)
        See :class:`BasicIterator`.
    max_instances_in_memory : int, optional, (default = None)
        See :class:`BasicIterator`.
    maximum_samples_per_batch : ``Tuple[str, int]``, (default = None)
        See :class:`BasicIterator`.
    skip_smaller_batches : bool, optional, (default = False)
        When the number of data samples is not dividable by `batch_size`,
        some batches might be smaller than `batch_size`.
        If set to `True`, those smaller batches will be discarded.
    """

    def __init__(self,
                 sorting_keys: List[Tuple[str, str]],
                 padding_noise: float = 0.1,
                 biggest_batch_first: bool = False,
                 batch_size: int = 32,
                 instances_per_epoch: int = None,
                 max_instances_in_memory: int = None,
                 cache_instances: bool = False,
                 track_epoch: bool = False,
                 proportional_sampling: bool = False,
                 maximum_samples_per_batch: Tuple[str, int] = None,
                 skip_smaller_batches: bool = False) -> None:
        if not sorting_keys:
            raise ConfigurationError("BucketIterator requires sorting_keys to be specified")

        super().__init__(cache_instances=cache_instances,
                         track_epoch=track_epoch,
                         batch_size=batch_size,
                         instances_per_epoch=instances_per_epoch,
                         max_instances_in_memory=max_instances_in_memory,
                         maximum_samples_per_batch=maximum_samples_per_batch)
        self._sorting_keys = sorting_keys
        self._padding_noise = padding_noise
        self._biggest_batch_first = biggest_batch_first
        self._skip_smaller_batches = skip_smaller_batches
        self._proportional_sampling = proportional_sampling

    @overrides
    def _create_batches(self, instances: Iterable[Instance], shuffle: bool) -> Iterable[Batch]:
        instances_by_data = group_by_dataset(instances)
        all_batches = []
        isTrain = False
        for instance in instances:
            isTrain = instance['metadata'].metadata['isTrain']
            break
        for instances in instances_by_data:
            for instance_list in self._memory_sized_lists(instances):

                instance_list = sort_by_padding(instance_list,
                                                self._sorting_keys,
                                                self.vocab,
                                                self._padding_noise)

                batches = []
                excess: Deque[Instance] = deque()
                for batch_instances in lazy_groups_of(iter(instance_list), self._batch_size):
                    for possibly_smaller_batches in self._ensure_batch_is_sufficiently_small(batch_instances, excess):
                        if self._skip_smaller_batches and len(possibly_smaller_batches) < self._batch_size:
                            continue
                        batches.append(Batch(possibly_smaller_batches))
                if excess and (not self._skip_smaller_batches or len(excess) == self._batch_size):
                    batches.append(Batch(excess))
                # TODO(brendanr): Add multi-GPU friendly grouping, i.e. group
                # num_gpu batches together, shuffle and then expand the groups.
                # This guards against imbalanced batches across GPUs.
                # TODO(ahmet): Look at biggest batch first thing..
                '''
                move_to_front = self._biggest_batch_first and len(batches) > 1
                if move_to_front:
                    # We'll actually pop the last _two_ batches, because the last one might not be full.
                    last_batch = batches.pop()
                    penultimate_batch = batches.pop()  
                '''
                if shuffle:
                    # NOTE: if shuffle is false, the data will still be in a different order
                    # because of the bucket sorting.
                    random.shuffle(batches)
                '''
                if move_to_front:
                    batches.insert(0, penultimate_batch)
                    batches.insert(0, last_batch)
                '''
            if self._proportional_sampling and isTrain:
                all_batches.append(batches)
            else:
                all_batches.extend(batches)
            
            #ROB: shuffle should always be true isnt it?, otherwise they will be ordered by task?
        if shuffle:
            # NOTE: if shuffle is false, the data will still be in a different order
            # because of the bucket sorting.
            if self._proportional_sampling:
                for i in range(len(all_batches)):
                    random.shuffle(all_batches[i])
            else:
                random.shuffle(all_batches)
        
        if self._proportional_sampling and isTrain:
            totalSize = sum([len(x) for x in all_batches])
            avgSize = totalSize / len(all_batches)
            totalBatches = avgSize * len(all_batches)
            print(totalSize)
            print(totalBatches)
            finishedDatasets = [False] * len(all_batches)
            batchIdxPerDataset = [0] * len(all_batches)
            counter = 0
            while counter < totalBatches:
            # for upsampling use the following instead:
            #while False in finishedDatasets:
                taskIdx = random.randint(0,len(all_batches)-1)
                if batchIdxPerDataset[taskIdx] == len(all_batches[taskIdx]) -1:
                    finishedDatasets[taskIdx] = True
                    batchIdxPerDataset[taskIdx] = 0
                counter += 1
                yield all_batches[taskIdx][batchIdxPerDataset[taskIdx]]
                batchIdxPerDataset[taskIdx] += 1
        else:
            for batch in all_batches:
                yield batch


