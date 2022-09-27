import copy
import datetime
import json
import logging
import os
import re
import unicodedata
from typing import List, Dict, Tuple, Optional, Any, Union, Iterator

import _jsonnet
import torch

logger = logging.getLogger(__name__)

from transformers import tokenization_utils

from machamp.data.machamp_instance import MachampInstance
from machamp.data.machamp_dataset import MachampDataset

ParameterGroupsType = List[Tuple[List[str], Dict[str, Any]]]


def load_json(path: str):
    """
    Loads a jsonnet file through the json package and returns a dict.
    
    Parameters
    ----------
    path: str
        the path to the json(net) file to load
    """
    return json.loads(_jsonnet.evaluate_snippet("", '\n'.join(open(path).readlines())))


def merge_configs(dataset_configs: List[str], parameters_config: Dict):
    """
    Merges the dataset configuration files, using also some default
    hyperparameters settings from the hyperparameter configuration.
    Note that we can have multiple dataset_configs, but each of them 
    can also have multiple datasets.

    Parameters
    ----------
    dataset_configs: List[str]
        List of paths to dataset configurations.
    parameters_config: Dict
        Path to the hyperparameters configuration.

    Returns
    ----------
    data_config: Dict
        The merged dataset configuration, where the first level of keys
        are the datasets to handle, and the configurations for each dataset
        are in the values.
    """
    data_params = {}
    for dataset_config_path in dataset_configs:
        config_file_data = load_json(dataset_config_path)
        for dataset in config_file_data:
            if dataset in data_params:
                logger.error("ERROR, same dataset identifier/name used twice: " + dataset)
                exit(1)
            data_params[dataset] = config_file_data[dataset]
            for task in data_params[dataset]['tasks']:
                task_type = data_params[dataset]['tasks'][task]['task_type']
                full_task_config = copy.deepcopy(parameters_config['decoders']['default_decoder'])
                full_task_config.update(parameters_config['decoders'][task_type])
                full_task_config.update(data_params[dataset]['tasks'][task])
                data_params[dataset]['tasks'][task] = full_task_config
    return data_params


def prep_batch(
        batch: List[MachampInstance],
        device: str,
        dataset: MachampDataset):
    """
    Converts a list of instances into a batch that can be used in the 
    forward pass of a MachampModel training. This means it converts a
    list of instances to a dictionary holding multiple torch.tensors, 
    containing at least the token_ids. Based on the setup it could also 
    conclude seg_ids, golds, offsets and mask.

    Parameters
    ----------
    batch: List[MachampInstance]
        A list of Machamp.data.MachampInstance . Where basically each instance
        represents a sentence.
    device: str
        Description of cuda device to use, i.e.: "cpu" or "gpu:0"
    dataset: MachampDataset
        Used for task-types.
        

    Returns
    -------
    batch: Dict[key: torch.tensor]
        'token_ids': Token ID's, looks like: [101, 20386, 19353, 102]
        'seg_ids': Indicate the segment id's, usually 0's, and can have 1's for other segments
        'golds': the gold annotation for the task, differs per task-type
        'offsets': The starting or ending index of each word based on wordpiece indices (note: is different length than token_ids)
        'subword_mask': The masking for the language model, shape=(batch_size, max_sent_len_subwords) filled with 1s and 0s. 
        'eval_mask': The masking for the evaluation. Is the length of the annotation.
    """
    batch_size = len(batch)
    max_subword_len = max([len(instance) for instance in batch])
    batch_tokens = torch.full((batch_size, max_subword_len), 0, dtype=torch.long, device=device)
    batch_seg_ids = torch.zeros((batch_size, max_subword_len), dtype=torch.long, device=device)
    golds = {}
    batch_offsets = None
    batch_eval_mask = None

    # Assuming here that batches are homogeneous, only checking
    # the first element.
    has_word_level = False
    for task in batch[0].golds:
        task_type = dataset.task_to_tasktype(task)
        if task_type in ['seq', 'multiseq', 'seq_bio', 'tok', 'dependency', 'string2string', 'mlm']:
            has_word_level = True

    if has_word_level:
        max_token_len = max([len(instance.offsets) for instance in batch])
        batch_offsets = torch.full((batch_size, max_token_len), -1, dtype=torch.long, device=device)
        batch_eval_mask = torch.zeros((batch_size, max_token_len), dtype=torch.bool, device=device)
    batch_subword_mask = torch.zeros((batch_size, max_subword_len), dtype=torch.bool, device=device)

    for task in batch[0].golds:
        task_type = dataset.task_to_tasktype(task)
        is_word_level = task_type in ['seq', 'multiseq', 'seq_bio', 'tok', 'dependency', 'string2string', 'mlm']

        if task_type == 'tok':
            golds[task] = torch.zeros((batch_size, max_subword_len - 2), dtype=torch.long, device=device)
        elif task_type == 'regression':
            golds[task] = torch.zeros(batch_size, dtype=torch.float, device=device)
        elif is_word_level:
            if len(batch[0].golds[task].shape) == 1:
                golds[task] = torch.zeros((batch_size, max_token_len), dtype=torch.long, device=device)
            else: # multiple annotations per token
                num_labels = len(dataset.vocabulary.get_vocab(task))
                golds[task] = torch.zeros((batch_size, max_token_len, num_labels), dtype=torch.long, device=device)
        elif task_type == 'multiclas':
            num_labels = len(dataset.vocabulary.get_vocab(task))
            golds[task] = torch.zeros(batch_size, num_labels, dtype=torch.long, device=device)
        else:
            golds[task] = torch.zeros(batch_size, dtype=torch.long, device=device)

    for instanceIdx, instance in enumerate(batch):
        batch_tokens[instanceIdx][0:len(instance.token_ids)] = instance.token_ids
        batch_seg_ids[instanceIdx][0:len(instance.seg_ids)] = instance.seg_ids
        for task in instance.golds:
            task_type = dataset.task_to_tasktype(task)
            is_word_level = task_type in ['seq', 'multiseq', 'seq_bio', 'tok', 'dependency', 'string2string', 'mlm']

            if is_word_level:
                if len(batch[0].golds[task].shape) == 1:
                    golds[task][instanceIdx][0:len(instance.golds[task])] = instance.golds[task]
                else:
                    for token_idx, token_labels in enumerate(instance.golds[task]):
                        for token_label in token_labels:
                            golds[task][instanceIdx][token_idx][token_label] = 1
            elif task_type == 'multiclas':
                for sent_label in instance.golds[task]:
                    golds[task][instanceIdx][sent_label] = 1
            else:
                golds[task][instanceIdx] = instance.golds[task]


        if has_word_level and type(batch[0].offsets) != type(None):
            batch_offsets[instanceIdx][:len(instance.offsets)] = instance.offsets
            batch_eval_mask[instanceIdx][:len(instance.offsets)] = 1
        batch_subword_mask[instanceIdx][:len(instance.token_ids)] = 1
    return {'token_ids': batch_tokens, 'seg_ids': batch_seg_ids, 'golds': golds, 'offsets': batch_offsets,
            'eval_mask': batch_eval_mask, 'subword_mask': batch_subword_mask}


def report_epoch(
        epoch_loss: float,
        dev_loss: float,
        epoch: int,
        train_metrics: Dict[str, float],
        dev_metrics: Dict[str, float],
        epoch_start_time: datetime.datetime,
        start_training_time: datetime.datetime, 
        device: str, 
        train_loss_dict: Dict[str, float],
        dev_loss_dict: Dict[str, float]):
    """
    Reports a variety of interesting and less interesting metrics that can
    be tracked across epochs. These are both logged and returned.

    Parameters
    ----------
    epoch_loss: float
        Loss on the training data.
    dev_loss: float
        Loss on the dev data.
    epoch: int
        The epoch we are currently on.
    train_metrics: Dict[str, float]
        All metrics based on the training data.
    dev_metrics: Dict[str, float]
        All metrics based on the dev data.
    epoch_start_time: datetime.datetime
        The time this epoch started.
    start_training_time: datetime.datetime
        The time the training procedure started.
    device: str
        Used to decide whether to print GPU ram
    train_loss_dict: Dict[str, float]
        training losses
    dev_loss_dict: Dict[str, float]
        dev losses

    Returns
    -------
    info: Dict[str, float]
        A dictionary containing all information that has just been logged
    """
    info = {'epoch': epoch}
    if 'cuda' in device:
        info['max_gpu_mem'] = torch.cuda.max_memory_allocated() * 1e-09

    _proc_status = '/proc/%d/status' % os.getpid()
    data = open(_proc_status).read()
    i = data.index('VmRSS:')
    info['cur_ram'] = int(data[i:].split(None, 3)[1]) * 1e-06

    # Might be nice to turn into a table?
    for task in train_loss_dict:
        info['train_' + task + '_loss'] = train_loss_dict[task]
    info['train_batch_loss'] = epoch_loss
    for metric in train_metrics:
        info['train_' + metric] = train_metrics[metric]
    for task in dev_loss_dict:
        info['dev_' + task + '_loss'] = dev_loss_dict[task]
    info['dev_batch_loss'] = dev_loss
    for metric in dev_metrics:
        info['dev_' + metric] = dev_metrics[metric]
    info['epoch_time'] = str(datetime.datetime.now() - epoch_start_time).split('.')[0]
    info['total_time'] = str(datetime.datetime.now() - start_training_time).split('.')[0]
    for key in info:
        if type(info[key]) == float:
            info[key] = '{:.4f}'.format(info[key])
    longest_key = max([len(key) for key in info]) + 1
    for key, value in info.items():
        logger.info(key + ' ' * (longest_key - len(key)) + ': ' + str(value))
    logger.info('\n')
    return info


def report_metrics(metrics: Dict[str, float]):
    """
    Reports evaluation metrics.

    Parameters
    ----------
    metrics: Dict[str, float]
        All metrics to report.

    Returns
    -------
    info: Dict[str, float]
        A dictionary containing all information that has just been logged
    """
    info = {}
    for metric in metrics:
        if type(metrics[metric]) == float:
            info[metric] = '{:.4f}'.format(metrics[metric])
        else:
            info[metric] = metrics[metric]
    longest_key = max([len(key) for key in info]) + 1
    for key, value in info.items():
        logger.info(key + ' ' * (longest_key - len(key)) + ': ' + str(value))
    logger.info('\n')
    return info


def clean_text(text: str):
    """
    Performs invalid character removal and whitespace cleanup on text. Based
    on _clean_text from BERT, but added unicode normalization and removal
    of double spaces.

    Parameters
    ----------
    text: str
        Input string (text).
    
    Returns
    -------
    cleaned_text: str
        The cleaned version of the input text.
    """
    cleaned_text = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xFFFD or tokenization_utils._is_control(char):
            continue
        if tokenization_utils._is_whitespace(char):
            cleaned_text.append(" ")
        else:
            cleaned_text.append(char)
    return "".join(cleaned_text).replace('  ', ' ')


# Taken from AllenNLP
def make_parameter_groups(
        model_parameters: Iterator[Tuple[str, torch.nn.Parameter]],
        groups: Optional[ParameterGroupsType] = None,
) -> Union[List[Dict[str, Any]], List[torch.nn.Parameter]]:
    """
    Takes a list of model parameters with associated names (typically coming from something like
    `model.named_parameters()`), along with a grouping (as specified below), and prepares them to be passed
    to the `__init__` function of a `torch.Optimizer`.  This means separating the parameters into
    groups with the given regexes, and prepping whatever keyword arguments are given for those
    regexes in `groups`.

    `groups` contains something like:

    ```
    [
        (["regex1", "regex2"], {"lr": 1e-3}),
        (["regex3"], {"lr": 1e-4})
    ]
    ```

    All of key-value pairs specified in each of these dictionaries will passed passed as-is
    to the optimizer, with the exception of a dictionaries that specify `requires_grad` to be `False`:

    ```
    [
        ...
        (["regex"], {"requires_grad": False})
    ]
    ```

    When a parameter group has `{"requires_grad": False}`, the gradient on all matching parameters
    will be disabled and that group will be dropped so that it's not actually passed to the optimizer.

    Ultimately, the return value of this function is in the right format to be passed directly
    as the `params` argument to a pytorch `Optimizer`.
    If there are multiple groups specified, this is a list of dictionaries, where each
    dict contains a "parameter group" and groups specific options, e.g., {'params': [list of
    parameters], 'lr': 1e-3, ...}.  Any config option not specified in the additional options (e.g.
    for the default group) is inherited from the top level arguments given in the constructor.  See:
    <https://pytorch.org/docs/0.3.0/optim.html?#per-parameter-options>.  See also our
    `test_optimizer_parameter_groups` test for an example of how this works in this code.

    The dictionary's return type is labeled as `Any`, because it can be a `List[torch.nn.Parameter]`
    (for the "params" key), or anything else (typically a float) for the other keys.
    """
    if groups:
        # In addition to any parameters that match group specific regex,
        # we also need a group for the remaining "default" group.
        # Those will be included in the last entry of parameter_groups.
        parameter_groups: Union[List[Dict[str, Any]], List[torch.nn.Parameter]] = [
            {"params": []} for _ in range(len(groups) + 1)
        ]
        # add the group specific kwargs
        for k in range(len(groups)):
            parameter_groups[k].update(groups[k][1])

        regex_use_counts: Dict[str, int] = {}
        parameter_group_names: List[set] = [set() for _ in range(len(groups) + 1)]
        for name, param in model_parameters:
            # Determine the group for this parameter.
            group_index = None
            for k, group_regexes in enumerate(groups):
                for regex in group_regexes[0]:
                    if regex not in regex_use_counts:
                        regex_use_counts[regex] = 0
                    if re.search(regex, name):
                        if group_index is not None and group_index != k:
                            raise ValueError(
                                "{} was specified in two separate parameter groups".format(name)
                            )
                        group_index = k
                        regex_use_counts[regex] += 1

            if group_index is not None:
                parameter_groups[group_index]["params"].append(param)
                parameter_group_names[group_index].add(name)
            else:
                # the default group
                parameter_groups[-1]["params"].append(param)
                parameter_group_names[-1].add(name)

        # find and remove any groups with 'requires_grad = False'
        no_grad_group_indices: List[int] = []
        for k, (names, group) in enumerate(zip(parameter_group_names, parameter_groups)):
            if group.get("requires_grad") is False:
                no_grad_group_indices.append(k)
                logger.info("Disabling gradient for the following parameters: %s", names)
                for param in group["params"]:
                    param.requires_grad_(False)

                # warn about any other unused options in that group.
                unused_options = {
                    key: val for key, val in group.items() if key not in ("params", "requires_grad")
                }
                if unused_options:
                    logger.warning("Ignoring unused options %s for %s", unused_options, names)
        parameter_group_names = [
            names
            for (k, names) in enumerate(parameter_group_names)
            if k not in no_grad_group_indices
        ]
        parameter_groups = [
            group for (k, group) in enumerate(parameter_groups) if k not in no_grad_group_indices
        ]

        # log the remaining parameter groups
        logger.info("Done constructing parameter groups.")
        for k in range(len(parameter_groups)):
            group_options = {
                key: val for key, val in parameter_groups[k].items() if key != "params"
            }
            logger.info("Group %s: %s, %s", k, list(parameter_group_names[k]), group_options)

        # check for unused regex
        for regex, count in regex_use_counts.items():
            if count == 0:
                logger.warning(
                    "When constructing parameter groups, %s does not match any parameter name",
                    regex,
                )
    else:
        parameter_groups = [param for name, param in model_parameters]

    # Log the number of parameters to optimize
    num_parameters = 0
    for parameter_group in parameter_groups:
        if isinstance(parameter_group, dict):
            num_parameters += sum(parameter.numel() for parameter in parameter_group["params"])
        else:
            num_parameters += parameter_group.numel()  # type: ignore
    logger.info("Number of trainable parameters: %s", num_parameters)

    return parameter_groups


class StreamToLogger(object):
    """
    Fake file-like stream object that redirects writes to a logger instance.
    """

    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass
