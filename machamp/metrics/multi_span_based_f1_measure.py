from collections import defaultdict
from typing import Dict, List, Optional, Set, Callable

import numpy
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.data.dataset_readers.dataset_utils.span_utils import TypedStringSpan
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics.metric import Metric
from torch.autograd import Variable

TAGS_TO_SPANS_FUNCTION_TYPE = Callable[[List[str], Optional[List[str]]], List[TypedStringSpan]]
# pylint: disable=invalid-name


def unwrap_to_tensors(*tensors: torch.Tensor):
    """
    If you actually passed gradient-tracking Tensors to a Metric, there will be
    a huge memory leak, because it will prevent garbage collection for the computation
    graph. This method ensures that you're using tensors directly and that they are on
    the CPU.
    """
    return (x.detach().cpu() if isinstance(x, torch.Tensor) else x for x in tensors)


@Metric.register("multi_span_f1")
class MultiSpanBasedF1Measure(Metric):
    """
    Extension of SpanBasedF1Measure; however naming is confusing, I do not think
    it does a span based thing anymore?, this should not be used as a metric to
    report, but should be fine as a proxy to pick the best model.
    """
    def __init__(self,
                 vocabulary: Vocabulary,
                 tag_namespace: str = "tags",
                 ignore_classes: List[str] = None,
                 label_encoding: Optional[str] = "BIO",
                 tags_to_spans_function: Optional[TAGS_TO_SPANS_FUNCTION_TYPE] = None,
                 threshold: float = 0.5,
                 max_heads: int = 2) -> None:
        """
        Parameters
        ----------
        vocabulary : ``Vocabulary``, required.
            A vocabulary containing the tag namespace.
        tag_namespace : str, required.
            This metric assumes that a BIO format is used in which the
            labels are of the format: ["B-LABEL", "I-LABEL"].
        ignore_classes : List[str], optional.
            Span labels which will be ignored when computing span metrics.
            A "span label" is the part that comes after the BIO label, so it
            would be "ARG1" for the tag "B-ARG1". For example by passing:

             ``ignore_classes=["V"]``
            the following sequence would not consider the "V" span at index (2, 3)
            when computing the precision, recall and F1 metrics.

            ["O", "O", "B-V", "I-V", "B-ARG1", "I-ARG1"]

            This is helpful for instance, to avoid computing metrics for "V"
            spans in a BIO tagging scheme which are typically not included.
        label_encoding : ``str``, optional (default = "BIO")
            The encoding used to specify label span endpoints in the sequence.
            Valid options are "BIO", "IOB1", "BIOUL" or "BMES".
        tags_to_spans_function: ``Callable``, optional (default = ``None``)
            If ``label_encoding`` is ``None``, ``tags_to_spans_function`` will be
            used to generate spans.
        threshold: threshold to decide how many labels to keep
        max_heads: maximum number of labels to predict per instance
        """
        if label_encoding and tags_to_spans_function:
            raise ConfigurationError(
                    'Both label_encoding and tags_to_spans_function are provided. '
                    'Set "label_encoding=None" explicitly to enable tags_to_spans_function.'
                    )
        if label_encoding:
            if label_encoding not in ["BIO", "IOB1", "BIOUL", "BMES"]:
                raise ConfigurationError("Unknown label encoding - expected 'BIO', 'IOB1', 'BIOUL', 'BMES'.")
        elif tags_to_spans_function is None:
            raise ConfigurationError(
                    'At least one of the (label_encoding, tags_to_spans_function) should be provided.'
                    )

        self._label_encoding = label_encoding
        self._tags_to_spans_function = tags_to_spans_function
        self._label_vocabulary = vocabulary.get_index_to_token_vocabulary(tag_namespace)
        self._ignore_classes: List[str] = ignore_classes or []

        # These will hold per label span counts.
        self._true_positives: Dict[str, int] = defaultdict(int)
        self._false_positives: Dict[str, int] = defaultdict(int)
        self._false_negatives: Dict[str, int] = defaultdict(int)

        self.vocabulary = vocabulary
        self.tag_namespace = tag_namespace

        self.threshold = threshold
        self.max_heads = max_heads

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor,
                 mask: Optional[torch.Tensor] = None,
                 prediction_map: Optional[torch.Tensor] = None):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, sequence_length, num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of integer class label of shape (batch_size, sequence_length). It must be the same
            shape as the ``predictions`` tensor without the ``num_classes`` dimension.
        mask: ``torch.Tensor``, optional (default = None).
            A masking tensor the same size as ``gold_labels``.
        prediction_map: ``torch.Tensor``, optional (default = None).
            A tensor of size (batch_size, num_classes) which provides a mapping from the index of predictions
            to the indices of the label vocabulary. If provided, the output label at each timestep will be
            ``vocabulary.get_index_to_token_vocabulary(prediction_map[batch, argmax(predictions[batch, t]))``,
            rather than simply ``vocabulary.get_index_to_token_vocabulary(argmax(predictions[batch, t]))``.
            This is useful in cases where each Instance in the dataset is associated with a different possible
            subset of labels from a large label-space (IE FrameNet, where each frame has a different set of
            possible roles associated with it).
        """
        if mask is None:
            mask = torch.ones_like(gold_labels)

        predictions, gold_labels, mask = unwrap_to_tensors(predictions, gold_labels, mask)

        if gold_labels.size() != predictions.size():
            raise ConfigurationError("Predictions and gold labels don't have the same size.")

        num_classes = predictions.size(-1)

        if (gold_labels >= num_classes).any():
            raise ConfigurationError("A gold label passed to MultiSpanBasedF1Measure contains an "
                                     "id >= {}, the number of classes.".format(num_classes))

        # Apply mask
        # Compute the mask before computing the loss
        # Transform the mask that is at the sentence level (#Size: n_batches x padded_document_length)
        # to a suitable format for the relation labels level
        _, padded_document_length, n_classes = predictions.size()
        mask = mask.float()
        squared_mask = mask.unsqueeze(-1).repeat(
            1, 1, num_classes
        )  # Size: n_batches x padded_document_length x padded_document_length x n_classes

        gold_labels = gold_labels.cpu()

        predictions = (
            predictions * squared_mask
        )  # Size: n_batches x padded_document_length x padded_document_length x n_classes
        gold_labels = (
            gold_labels * squared_mask
        )  # Size: n_batches x padded_document_length x padded_document_length x n_classes

        outside_index = self.vocabulary.get_token_index("O", namespace=self.tag_namespace)

        thresh = self.threshold
        t = Variable(torch.Tensor([thresh]))
        pred_over_thresh = (predictions >= t).float() * 1

        # @AR: Get the thresholded matrix and prepare the prediction sequence
        maxxx_ = numpy.argmax(predictions, axis=-1).tolist()
        maxxx = torch.nn.functional.one_hot(torch.tensor(maxxx_), num_classes)

        # @AR: For each label set, check if to apply argmax or sigmoid thresh
        for i in range(gold_labels.size(0)):  # batch_size
            j = 0
            for pred in pred_over_thresh[i]:
                num_pred_over_thresh = numpy.count_nonzero(pred)

                if (num_pred_over_thresh == 0) or (num_pred_over_thresh == 1):
                    if numpy.count_nonzero(predictions[i][j]) == 0:  # if it is a pad, put all zeros
                        predictions[i][j] = torch.tensor([0.]*num_classes)
                    else:
                        pred_idx_list = maxxx[i][j]
                        predictions[i][j] = pred_idx_list

                elif num_pred_over_thresh <= self.max_heads:
                    pred_idx_list = numpy.argpartition(pred, -num_pred_over_thresh)[-num_pred_over_thresh:]

                    try:
                        outside_position = pred_idx_list.tolist().index(outside_index)
                    except ValueError:
                        outside_position = -1
                    # for el_i in range(len(pred_idx_list)):
                    #     if pred_idx_list[el_i] == outside_index:
                    #         outside_position = el_i
                    #         break
                    if outside_position != -1:
                        pred_len = len(pred_idx_list)-1
                        # If the last (i.e., the best) is "O", ignore/remove the others
                        if outside_position == pred_len:
                            pred_idx_list = [pred_idx_list[-1]]
                        # O.w. get only from the last before the "O"
                        else:
                            # del pred_idx_list[outside_position]
                            pred_idx_list = pred_idx_list[outside_position+1:]

                    multi_one_hot = []
                    for index in range(num_classes):
                        if index in pred_idx_list:
                            multi_one_hot.append(1.)
                        else:
                            multi_one_hot.append(0.)

                    predictions[i][j] = torch.tensor(multi_one_hot)

                else:
                    pred_idx_list = numpy.argpartition(pred, -self.max_heads)[-self.max_heads:]

                    try:
                        outside_position = pred_idx_list.tolist().index(outside_index)
                    except ValueError:
                        outside_position = -1
                    # outside_position = None
                    # for el_i in range(len(pred_idx_list)):
                    #     if pred_idx_list[el_i] == outside_index:
                    #         outside_position = el_i
                    #         break
                    if outside_position != -1:
                        pred_len = len(pred_idx_list)-1
                        # If the last (i.e., the best) is "O", ignore/remove the others
                        if outside_position == pred_len:
                            pred_idx_list = [pred_idx_list[-1]]
                        # O.w. get only from the last before the "O"
                        else:
                            # del pred_idx_list[outside_position]
                            pred_idx_list = pred_idx_list[outside_position+1:]

                    multi_one_hot = []
                    for index in range(num_classes):
                        if index in pred_idx_list:
                            multi_one_hot.append(1.)
                        else:
                            multi_one_hot.append(0.)

                    predictions[i][j] = torch.tensor(multi_one_hot)
                j += 1

        # Iterate over timesteps in batch.
        batch_size = gold_labels.size(0)
        for i in range(batch_size):
            # flattened_predictions = predictions[i].nonzero().cpu().numpy()
            # flattened_gold_labels = gold_labels[i].nonzero().cpu().numpy()

            for j in range(len(predictions[i])):
                preds = predictions[i][j].nonzero().cpu().numpy()
                golds = gold_labels[i][j].nonzero().cpu().numpy()

                for prediction in preds:
                    if prediction in golds:
                        if prediction != outside_index:
                            self._true_positives[str(prediction[0])] += 1
                    else:
                        self._false_positives[str(prediction[0])] += 1
                for gold in golds:
                    if gold not in preds:
                        self._false_negatives[str(gold[0])] += 1

    def get_metric(self, reset: bool = False):
        """
        Returns
        -------
        A Dict per label containing following the span based metrics:
        precision : float
        recall : float
        f1-measure : float

        Additionally, an ``overall`` key is included, which provides the precision,
        recall and f1-measure for all spans.
        """
        all_tags: Set[str] = set()
        all_tags.update(self._true_positives.keys())
        all_tags.update(self._false_positives.keys())
        all_tags.update(self._false_negatives.keys())
        all_metrics = {}
        for tag in all_tags:
            precision, recall, f1_measure = self._compute_metrics(self._true_positives[tag],
                                                                  self._false_positives[tag],
                                                                  self._false_negatives[tag])
            precision_key = "precision" + "-" + tag
            recall_key = "recall" + "-" + tag
            f1_key = "f1-measure" + "-" + tag
            all_metrics[precision_key] = precision
            all_metrics[recall_key] = recall
            all_metrics[f1_key] = f1_measure
        # Compute the precision, recall and f1 for all spans jointly.
        precision, recall, f1_measure = self._compute_metrics(sum(self._true_positives.values()),
                                                              sum(self._false_positives.values()),
                                                              sum(self._false_negatives.values()))
        all_metrics["precision-overall"] = precision
        all_metrics["recall-overall"] = recall
        all_metrics["f1-measure-overall"] = f1_measure
        if reset:
            self.reset()
        return all_metrics

    @staticmethod
    def _compute_metrics(true_positives: int, false_positives: int, false_negatives: int):
        precision = float(true_positives) / float(true_positives + false_positives + 1e-13)
        recall = float(true_positives) / float(true_positives + false_negatives + 1e-13)
        f1_measure = 2. * ((precision * recall) / (precision + recall + 1e-13))
        return precision, recall, f1_measure

    def reset(self):
        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)
