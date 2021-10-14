import logging
import math
import os
import random
from typing import List, Iterable, Tuple, Iterator, TypeVar
from itertools import islice, zip_longest


from allennlp.common.checks import ConfigurationError
#from allennlp.common.util import lazy_groups_of
from allennlp.data.instance import Instance
from allennlp.data.samplers import BatchSampler
from torch.utils import data

logger = logging.getLogger(__name__)


A = TypeVar("A")

def add_noise_to_value(value: int, noise_param: float):
    noise_value = value * noise_param
    noise = random.uniform(-noise_value, noise_value)
    return value + noise

def lazy_groups_of(indices: Iterable[A], group_size: int, sizes: Iterable[A], batch_size, max_tokens) -> Iterator[List[A]]:
    """
    Takes an iterable and batches the individual instances into lists of the
    specified size. The last list may be smaller if there are instances left over.
    """
    cur_batch = []
    cur_size = 0
    for indice, size in zip(indices, sizes):
        # If the batch is saturated or the max #tokens is reached, return the batch and restart
        if (len(cur_batch) == batch_size and batch_size != -1) or (max_tokens != -1 and cur_size + size[0] > max_tokens):
            yield cur_batch
            cur_batch = [indice]
            cur_size = size[0]
        # O.w., add the example to the batch and keep track of the num of tokens
        else:
            cur_batch.append(indice)
            cur_size += size[0]
    # Add the last batch, even if it is smaller
    if len(cur_batch) > 0:
        yield cur_batch

@BatchSampler.register("dataset_buckets")
class BucketBatchSampler(BatchSampler):
    """
    Based on the BucketSampler, but makes sure each batch is from one dataset
    (homogeneous). Also supports some sampling strategies.

    # Parameters

    data_source: `data.Dataset`, required
        The pytorch `Dataset` of allennlp Instances to bucket.

        In a typical AllenNLP configuration file, this parameter does not get an entry under the
        "batch_sampler", it gets constructed separately.
    batch_size : `int`, required
        The size of each batch of instances yielded when calling the dataloader.

    sorting_keys : `List[str]`, optional
        To bucket inputs into batches, we want to group the instances by padding length, so that we
        minimize the amount of padding necessary per batch. In order to do this, we need to know
        which fields need what type of padding, and in what order.

        Specifying the right keys for this is a bit cryptic, so if this is not given we try to
        auto-detect the right keys by iterating through a few instances upfront, reading all of the
        padding keys and seeing which one has the longest length.  We use that one for padding.
        This should give reasonable results in most cases. Some cases where it might not be the
        right thing to do are when you have a `ListField[TextField]`, or when you have a really
        long, constant length `ArrayField`.

        When you need to specify this yourself, you can create an instance from your dataset and
        call `Instance.get_padding_lengths()` to see a list of all keys used in your data.  You
        should give one or more of those as the sorting keys here.

    padding_noise : `float`, optional (default=`.1`)
        When sorting by padding length, we add a bit of noise to the lengths, so that the sorting
        isn't deterministic.  This parameter determines how much noise we add, as a percentage of
        the actual padding value for each instance.

    drop_last : `bool`, (default = `False`)
        If `True`, the sampler will drop the last batch if
        its size would be less than batch_size`.

    """

    def __init__(
        self,
        data_source: data.Dataset,
        batch_size: int,
        sorting_keys: List[str] = None,
        padding_noise: float = 0.1,
        drop_last: bool = False,
        sampling_smoothing: float = 1.0,
        max_tokens: int = -1,
    ):

        self.vocab = data_source.vocab
        self.sorting_keys = sorting_keys
        self.padding_noise = padding_noise
        self.batch_size = batch_size
        self.data_source = data_source
        self.drop_last = drop_last
        self.sampling_smoothing = sampling_smoothing
        self.first = True
        self.max_tokens = max_tokens

    def _argsort_by_padding(
        self, instances: Iterable[Instance], indices: Iterable[int]
    ) -> Tuple[List[int], List[List[int]]]:
        """
        Argsorts the instances by their padding lengths, using the keys in
        `sorting_keys` (in the order in which they are provided). `sorting_keys`
        is a list of `(field_name, padding_key)` tuples.
        """
        if not self.sorting_keys:
            logger.info("No sorting keys given; trying to guess a good one")
            self._guess_sorting_keys(instances)
            logger.info(f"Using {self.sorting_keys} as the sorting keys")

        instances_with_lengths = []
        for instance in instances:
            # Make sure instance is indexed before calling .get_padding
            lengths = []
            noisy_lengths = []
            for field_name in self.sorting_keys:
                if field_name not in instance.fields:
                    raise ConfigurationError(
                        f'Sorting key "{field_name}" is not a field in instance. '
                        f"Available fields/keys are {list(instance.fields.keys())}."
                    )
                lengths.append(len(instance.fields[field_name]))
                noisy_lengths.append(add_noise_to_value(lengths[-1], self.padding_noise))
            instances_with_lengths.append((noisy_lengths, lengths, instance))

        with_indices = [(x, i) for i, x in zip(indices, instances_with_lengths)]
        with_indices.sort(key=lambda x: x[0][0])
        return (
            [instance_with_index[-1] for instance_with_index in with_indices],
            [instance_with_index[0][1] for instance_with_index in with_indices],
        )

    def group_by_dataset(self, instances: List[Instance]) -> Tuple[List[int], List[Instance]]:
        # returns list of tuples; one tuple for each dataset
        # tuple contains list of instances with their index in the full instances list
        by_datasets = {}
        for instance_idx, instance in enumerate(instances):
            dataset = instance.fields['dataset'].label
            if dataset not in by_datasets:
                by_datasets[dataset] = ([], [])
            by_datasets[dataset][0].append(instance_idx)
            by_datasets[dataset][1].append(instance)
        for dataset in by_datasets:
            yield by_datasets[dataset]
        
    def __iter__(self) -> Iterable[List[int]]:
        if self.first and os.path.isfile('docs/machamp.txt'):
            champ_txt = "\nMaChAmp succesfully initialized\n"
            for line in open('docs/machamp.txt'):
                champ_txt += line
            logger.info(champ_txt)
            self.first = False
        
        is_train = True
        for instance in self.data_source.instances:
            is_train = instance['metadata'].metadata['is_train']
            break

        all_batches = []
        for dataset_indices, dataset_instances in self.group_by_dataset(self.data_source.instances):
            # Rob: it now passes a list of instances as well as indices
            # because for the 2-n datasets, the indices shouldnt start at 0
            sorted_indices, lengths = self._argsort_by_padding(dataset_instances, dataset_indices)
            dataset_batches = []
            for group in lazy_groups_of(sorted_indices, self.batch_size, lengths, self.batch_size, self.max_tokens):
                batch_indices = list(group)
                if self.drop_last and len(batch_indices) < self.batch_size:
                    continue
                dataset_batches.append(batch_indices)

            # for smoothing the dataset sizes we need to keep the datasets separated
            if self.sampling_smoothing != 1.0 and is_train: # TODO also smooth during dev?, but not during prediction!
                all_batches.append(dataset_batches)
            # without smoothing, they are all put into one long list
            else:
                all_batches.extend(dataset_batches)

        if self.sampling_smoothing != 1.0 and is_train:
            # calculate new size based on smoothing
            sizes = [len(x) for x in all_batches]
            new_sizes = []
            total_size = sum(sizes)
            total_new_prob = 0.0
            for size in sizes:
                pi = size/total_size
                total_new_prob += math.pow(pi, self.sampling_smoothing)
            
            for size in sizes:
                pi = size/total_size
                prob = (1/pi) * (math.pow(pi, self.sampling_smoothing)/total_new_prob)
                new_sizes.append(int(size * prob))

            # collect all batches
            this_epoch_all_batches = []
            for dataset_idx in range(len(all_batches)):
                new_size = new_sizes[dataset_idx]
                #random.shuffle(all_batches[dataset_idx])
                while new_size > len(all_batches[dataset_idx]):
                    all_batches[dataset_idx] += all_batches[dataset_idx]
                this_epoch_all_batches += all_batches[dataset_idx][:new_size]

            # shuffle all batches
            random.shuffle(this_epoch_all_batches)
            for batch in this_epoch_all_batches:
                if len(batch) > 0:
                    yield batch

        else:
            random.shuffle(all_batches)
            for batch in all_batches:
                if len(batch) > 0:
                    yield batch
        
    def _guess_sorting_keys(self, instances: Iterable[Instance], num_instances: int = 10) -> None:
        """
        Use `num_instances` instances from the dataset to infer the keys used
        for sorting the dataset for bucketing.

        # Parameters

        instances : `Iterable[Instance]`, required.
            The dataset to guess sorting keys for.
        num_instances : `int`, optional (default = `10`)
            The number of instances to use to guess sorting keys. Typically
            the default value is completely sufficient, but if your instances
            are not homogeneous, you might need more.
        """
        max_length = 0.0
        longest_field: str = None
        for i, instance in enumerate(instances):
            instance.index_fields(self.vocab)
            for field_name, field in instance.fields.items():
                length = len(field)
                if length > max_length:
                    max_length = length
                    longest_field = field_name
            if i > num_instances:
                # Only use num_instances instances to guess the sorting keys.
                break

        if not longest_field:
            # This shouldn't ever happen (you basically have to have an empty instance list), but
            # just in case...
            raise AssertionError(
                "Found no field that needed padding; we are surprised you got this error, please "
                "open an issue on github"
            )
        self.sorting_keys = [longest_field]

    def __len__(self):
        batch_count_float = len(self.data_source) / self.batch_size
        if self.drop_last:
            return math.floor(batch_count_float)
        else:
            return math.ceil(batch_count_float)
