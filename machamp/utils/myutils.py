import copy
import json
import logging
import re
from typing import List, Dict, Tuple, Optional, Any, Union, Iterator

import _jsonnet
import torch

logger = logging.getLogger(__name__)

from transformers import tokenization_utils
from transformers import AutoTokenizer

from machamp.data.machamp_instance import MachampInstance
from machamp.data.machamp_dataset import MachampDataset
from machamp.modules.allennlp.scalar_mix import ScalarMix

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
        dataset: MachampDataset,
        assume_word_level: bool = False):
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
    assume_word_level: bool
        Normally, we check the gold annotations to see whether we need word
        level information (i.e. offsets); but if gold data is absent, and we
        still need to perform token level tasks, this variable can enforce
        getting the word level information.
        

    Returns
    -------
    batch: Dict[key: torch.tensor]
        'token_ids': Token ID's, looks like: [101, 20386, 19353, 102]
        'seg_ids': Indicate the segment id's, usually 0's, and can have 1's for other segments
        'golds': the gold annotation for the task, differs per task-type
        'offsets': The starting or ending index of each word based on wordpiece indices (note: is different length than
        token_ids)
        'subword_mask': The masking for the language model, shape=(batch_size, max_sent_len_subwords) filled with 1s
        and 0s.
        'task_masks': multiseq, multiclas and seq_bio need a word-level mask. This is for evaluation purposes mainly.
    """
    batch_size = len(batch)
    max_subword_len = max([len(instance) for instance in batch])
    batch_tokens = torch.full((batch_size, max_subword_len), 0, dtype=torch.long, device=device)
    batch_seg_ids = torch.zeros((batch_size, max_subword_len), dtype=torch.long, device=device)
    golds = {}
    batch_offsets = None
    batch_word_mask = None

    # Check if any of the task is token level, because we need to save the offsets and a 
    # separate mask
    all_tasks = set()
    has_word_level = assume_word_level
    for instance in batch:
        for task in instance.golds:
            all_tasks.add(task)
            task_type = dataset.task_to_tasktype(task)
            if task_type in ['seq', 'multiseq', 'seq_bio', 'tok', 'dependency', 'string2string', 'mlm']:
                has_word_level = True
    if has_word_level:
        max_token_len = max([len(instance.offsets) for instance in batch if type(instance.offsets) != type(None)])
        batch_offsets = torch.full((batch_size, max_token_len), -1, dtype=torch.long, device=device)
        batch_word_mask = torch.zeros((batch_size, max_token_len), dtype=torch.bool, device=device)

    batch_subword_mask = torch.zeros((batch_size, max_subword_len), dtype=torch.bool, device=device)
    task_masks = {}
    for task in all_tasks:
        task_type = dataset.task_to_tasktype(task)

        if task_type == 'tok':
            golds[task] = torch.full((batch_size, max_subword_len - 2), -100, dtype=torch.long, device=device)
        elif task_type == 'regression':
            golds[task] = torch.full((batch_size,), -100, dtype=torch.float, device=device)
        elif task_type == 'multiseq':
            num_labels = len(dataset.vocabulary.get_vocab(task))
            golds[task] = torch.full((batch_size, max_token_len, num_labels), -100, dtype=torch.long, device=device)
        elif task_type == 'multiclas':
            num_labels = len(dataset.vocabulary.get_vocab(task))
            golds[task] = torch.full((batch_size, num_labels), -100, dtype=torch.long, device=device)
        elif task_type == 'classification':
            golds[task] = torch.full((batch_size,), -100, dtype=torch.long, device=device)
        else: # token level task
            golds[task] = torch.full((batch_size, max_token_len), -100, dtype=torch.long, device=device)
        if task_type == 'dependency':
            if task.endswith('-rels'):
                task_masks[task[:-5]] = torch.zeros((batch_size), dtype=torch.bool, device=device)
        else:
            task_masks[task] = torch.zeros((batch_size), dtype=torch.bool, device=device)

    for instanceIdx, instance in enumerate(batch):
        batch_tokens[instanceIdx][0:len(instance.token_ids)] = instance.token_ids
        batch_seg_ids[instanceIdx][0:len(instance.seg_ids)] = instance.seg_ids
        for task in instance.golds:
            task_type = dataset.task_to_tasktype(task)

            if task_type == 'multiseq':
                for token_idx, token_labels in enumerate(instance.golds[task]):
                    for token_label in token_labels:
                        if token_label != -100:
                            golds[task][instanceIdx][token_idx][token_label] = 1
            elif task_type == 'multiclas':
                for sent_label in instance.golds[task]:
                    if sent_label != -100:
                        golds[task][instanceIdx][sent_label] = 1
            elif task_type in ['regression', 'classification']:
                golds[task][instanceIdx] = instance.golds[task]
            else: # token level task
                golds[task][instanceIdx][0:len(instance.golds[task])] = instance.golds[task]
            
            if task_type == 'dependency':
                if task.endswith('-rels'):
                    task_masks[task[:-5]][instanceIdx] = True
            else:
                task_masks[task][instanceIdx] = True

        if has_word_level and type(instance.offsets) != type(None):
            batch_offsets[instanceIdx][:len(instance.offsets)] = instance.offsets
            batch_word_mask[instanceIdx][:len(instance.offsets)] = 1
        batch_subword_mask[instanceIdx][:len(instance.token_ids)] = 1

    return {'token_ids': batch_tokens, 'seg_ids': batch_seg_ids, 'golds': golds, 'offsets': batch_offsets,
            'subword_mask': batch_subword_mask, 'task_masks': task_masks, 'word_mask': batch_word_mask}


def report_metrics(metrics):
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
        if metric == 'sum':
            info[metric] = '{:.4f}'.format(metrics[metric])
            continue
        optim_metric = metrics[metric]['optimization_metrics']
        value = metrics[metric][optim_metric][optim_metric]
        if type(value) == float:
            info[metric + '_' + optim_metric] = '{:.4f}'.format(value)
        else:
            info[metric + '_' + optim_metric] = value
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


def apply_scalar(mlm_out: torch.tensor, layers: List, scalar: ScalarMix):
    """
    Applies attention to the layers of the output of the LM. 
    If the number of layers is 1, no attention is necessary. 

    Parameters
    ----------
    mlm_out: torch.tensor
        Input, shape = [layers, batch_size, num_tokens, emb_size]
    layers: List
        Which layers we should use (indices)
    scalar: ScalarMix):
        The scalar to apply

    Returns
    -------
    result: torch.tensor
        Shape should equal the input, but then without the first dimension
    """
    if len(layers) > 1:
        return scalar(mlm_out[layers])
    else:
        return mlm_out[layers[0]]


def identify_tokenizer(tokenizer: AutoTokenizer):
    """
    Identifies the strategy the tokenizer uses to represent (the absence of) whitespaces. 
    Could be one of 'wordpiece', 'sentencepiece' or 'other'. Note that some have no special
    characters for this, I am not sure what to do with those yet (tokenization should be easier?)
    other exceptions are xlm-mlm-100-1280 is special, it has </w> and sberbank-ai/mGPT 
    has a Ġ for whitespaces.

    Parameters
    ----------
    tokenizer: AutoTokenizer
        the tokenizer to inspect
    
    Returns
    -------
    tokenizer_type: str
        the type of the tokenizer, one of 'wordpiece', 'sentencepiece' or 'other'
    """
    result = ''.join(tokenizer.tokenize('test testestest'))
    if '##' in result:
        return 'wordpiece'
    elif '▁' in result:
        return 'sentencepiece'
    else:
        return 'other'



class StreamToLogger2(object):
   """
   Fake file-like stream object that redirects writes to a logger instance.
   From: https://stackoverflow.com/questions/19425736/how-to-redirect-stdout-and-stderr-to-logger-in-python
   """
   def __init__(self, logger, log_level=logging.INFO):
      self.logger = logger
      self.log_level = log_level
      self.linebuf = ''
 
   def write(self, buf):
      for line in buf.rstrip().splitlines():
         self.logger.log(self.log_level, line.rstrip())

   def flush(self):
        pass

