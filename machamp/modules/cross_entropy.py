import copy
import json
import logging
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union

import math
import numpy
import torch

from allennlp.common.checks import ConfigurationError

logger = logging.getLogger(__name__)

T = TypeVar("T")



def sequence_cross_entropy_with_logits(
    logits: torch.FloatTensor,
    targets: torch.LongTensor,
    weights: Union[torch.FloatTensor, torch.BoolTensor],
    average: str = "batch",
    label_smoothing: float = None,
    gamma: float = None,
    alpha: Union[float, List[float], torch.FloatTensor] = None,
) -> torch.FloatTensor:
    """
    Computes the cross entropy loss of a sequence, weighted with respect to
    some user provided weights. Note that the weighting here is not the same as
    in the `torch.nn.CrossEntropyLoss()` criterion, which is weighting
    classes; here we are weighting the loss contribution from particular elements
    in the sequence. This allows loss computations for models which use padding.

    # Parameters

    logits : `torch.FloatTensor`, required.
        A `torch.FloatTensor` of size (batch_size, sequence_length, num_classes)
        which contains the unnormalized probability for each class.
    targets : `torch.LongTensor`, required.
        A `torch.LongTensor` of size (batch, sequence_length) which contains the
        index of the true class for each corresponding step.
    weights : `Union[torch.FloatTensor, torch.BoolTensor]`, required.
        A `torch.FloatTensor` of size (batch, sequence_length)
    average: `str`, optional (default = `"batch"`)
        If "batch", average the loss across the batches. If "token", average
        the loss across each item in the input. If `None`, return a vector
        of losses per batch element.
    label_smoothing : `float`, optional (default = `None`)
        Whether or not to apply label smoothing to the cross-entropy loss.
        For example, with a label smoothing value of 0.2, a 4 class classification
        target would look like `[0.05, 0.05, 0.85, 0.05]` if the 3rd class was
        the correct label.
    gamma : `float`, optional (default = `None`)
        Focal loss[*] focusing parameter `gamma` to reduces the relative loss for
        well-classified examples and put more focus on hard. The greater value
        `gamma` is, the more focus on hard examples.
    alpha : `Union[float, List[float]]`, optional (default = `None`)
        Focal loss[*] weighting factor `alpha` to balance between classes. Can be
        used independently with `gamma`. If a single `float` is provided, it
        is assumed binary case using `alpha` and `1 - alpha` for positive and
        negative respectively. If a list of `float` is provided, with the same
        length as the number of classes, the weights will match the classes.
        [*] T. Lin, P. Goyal, R. Girshick, K. He and P. Dollár, "Focal Loss for
        Dense Object Detection," 2017 IEEE International Conference on Computer
        Vision (ICCV), Venice, 2017, pp. 2999-3007.

    # Returns

    `torch.FloatTensor`
        A torch.FloatTensor representing the cross entropy loss.
        If `average=="batch"` or `average=="token"`, the returned loss is a scalar.
        If `average is None`, the returned loss is a vector of shape (batch_size,).

    """
    if average not in {None, "token", "batch"}:
        raise ValueError("Got average f{average}, expected one of None, 'token', or 'batch'")


    lss = torch.nn.KLDivLoss()(logits,targets)

    return lss


