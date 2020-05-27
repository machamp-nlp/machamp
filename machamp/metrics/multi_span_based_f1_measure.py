from typing import Dict, List, Optional, Set, Callable
from collections import defaultdict

import numpy
import torch
from torch.autograd import Variable

from allennlp.common.checks import ConfigurationError
from allennlp.nn.util import get_lengths_from_binary_sequence_mask
from allennlp.data.vocabulary import Vocabulary
from allennlp.training.metrics.metric import Metric
from allennlp.data.dataset_readers.dataset_utils.span_utils import (
        bio_tags_to_spans,
        bioul_tags_to_spans,
        iob1_tags_to_spans,
        bmes_tags_to_spans,
        TypedStringSpan
)

from machamp.util import to_multilabel_sequence

TAGS_TO_SPANS_FUNCTION_TYPE = Callable[[List[str], Optional[List[str]]], List[TypedStringSpan]]  # pylint: disable=invalid-name


@Metric.register("multi_span_f1")
class MultiSpanBasedF1Measure(Metric):
    """
    The Conll SRL metrics are based on exact span matching. This metric
    implements span-based precision and recall metrics for a BIO tagging
    scheme. It will produce precision, recall and F1 measures per tag, as
    well as overall statistics. Note that the implementation of this metric
    is not exactly the same as the perl script used to evaluate the CONLL 2005
    data - particularly, it does not consider continuations or reference spans
    as constituents of the original span. However, it is a close proxy, which
    can be helpful for judging model performance during training. This metric
    works properly when the spans are unlabeled (i.e., your labels are
    simply "B", "I", "O" if using the "BIO" label encoding).

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

        #mask = mask.unsqueeze(2).expand(mask.shape[0], mask.shape[1], predictions.size(-1))

        #predictions = torch.round(predictions)

        #predictions, gold_labels, mask, prediction_map = self.unwrap_to_tensors(predictions,
        predictions, gold_labels, mask = self.unwrap_to_tensors(predictions,
                                                                                gold_labels,
                                                                                mask)#, prediction_map)

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
        #squared_mask = torch.stack([e.view(padded_document_length, 1) * e for e in mask], dim=0)
        squared_mask = mask.unsqueeze(-1).repeat(
        #squared_mask = squared_mask.unsqueeze(-1).repeat(
            1, 1, num_classes
            #1, 1, 1, n_classes
        )  # Size: n_batches x padded_document_length x padded_document_length x n_classes
        
        #print(predictions.size()) # 2x21x5
        #print(squared_mask.size()) #2x21x5


        # HMTL
        gold_labels = gold_labels.cpu()

        predictions = (
            predictions * squared_mask
        )  # Size: n_batches x padded_document_length x padded_document_length x n_classes
        gold_labels = (
            gold_labels * squared_mask
        )  # Size: n_batches x padded_document_length x padded_document_length x n_classes

        #print(predictions[0]) #21x5
        #argmax_predictions = predictions.max(-1)[1]
        #import torch.nn.functional as F
        #_hot = F.one_hot(argmax_predictions, num_classes=num_classes)
        #predictions = _hot

        outside_index = self.vocabulary.get_token_index("O", namespace=self.tag_namespace)

        thresh = self.threshold
        t = Variable(torch.Tensor([thresh]))
        pred_over_thresh = (predictions >= t).float() * 1

        # ==========

        # @AR: Get the thresholded matrix and prepare the prediction sequence
        maxxx_ = numpy.argmax(predictions, axis=-1).tolist()
        maxxx = torch.nn.functional.one_hot(torch.tensor(maxxx_), num_classes)

        # @AR: For each label set, check if to apply argmax or sigmoid thresh
        for i in range(gold_labels.size(0)): # batch_size
            j=0
            for pred in pred_over_thresh[i]:
                num_pred_over_thresh = numpy.count_nonzero(pred)

                if (num_pred_over_thresh == 0) or (num_pred_over_thresh == 1):
                    if numpy.count_nonzero(predictions[i][j]) == 0: # if it is a pad, put all zeros
                        predictions[i][j] = torch.tensor([0.]*num_classes)
                    else:
                        pred_idx_list = maxxx[i][j]
                        predictions[i][j] = pred_idx_list

                elif num_pred_over_thresh <= self.max_heads:
                    pred_idx_list = numpy.argpartition(pred, -num_pred_over_thresh)[-num_pred_over_thresh:]

                    outside_position = -1
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

                    #print("here NOT")
                    #print(pred_idx_list.tolist())
                    multi_one_hot = []
                    for index in range(num_classes):
                        if index in pred_idx_list:
                            multi_one_hot.append(1.)
                        else:
                            multi_one_hot.append(0.)

                    predictions[i][j] = torch.tensor(multi_one_hot)

                else:
                    pred_idx_list = numpy.argpartition(pred, -self.max_heads)[-self.max_heads:]
                    # # print("sigmoid ->", pred_idx_list)

                    outside_position = -1
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

                    #print("here NOT")
                    #print(pred_idx_list.tolist())
                    multi_one_hot = []
                    for index in range(num_classes):
                        if index in pred_idx_list:
                            multi_one_hot.append(1.)
                        else:
                            multi_one_hot.append(0.)

                    predictions[i][j] = torch.tensor(multi_one_hot)







                # if num_pred_over_thresh < self.max_heads:
                #     if numpy.count_nonzero(predictions[i][j]) == 0: # if it is a pad, put all zeros
                #         predictions[i][j] = torch.tensor([0.]*num_classes)
                #     else:
                #         pred_idx_list = maxxx[i][j]
                #         predictions[i][j] = pred_idx_list
                #     # print("argmax  ->", pred_idx_list)
                # else:
                #     #pred_idx_list = [maxxx[j]]
                #     pred_idx_list = numpy.argpartition(pred, -self.max_heads)[-self.max_heads:]
                #     # # print("sigmoid ->", pred_idx_list)

                #     # # If the first (i.e., second best) is "O", ignore/remove it
                #     if pred_idx_list[0] == outside_index:
                #         pred_idx_list = pred_idx_list[1:]
                #     # If the second (i.e., the best) is "O", ignore/remove the first
                #     elif pred_idx_list[1] == outside_index:
                #         pred_idx_list = pred_idx_list[1:]
                #     else:
                #         pass

                #     #print("here NOT")
                #     #print(pred_idx_list.tolist())
                #     multi_one_hot = []
                #     for index in range(num_classes):
                #         if index in pred_idx_list:
                #             multi_one_hot.append(1.)
                #         else:
                #             multi_one_hot.append(0.)

                #     predictions[i][j] = torch.tensor(multi_one_hot)

                j += 1

        # ==========


        #predictions = pred_over_thresh
        #print(pred_over_thresh, pred_over_thresh.shape, type(pred_over_thresh))
        #print(predictions, predictions.shape, type(predictions))



        # Iterate over timesteps in batch.
        batch_size = gold_labels.size(0)
        for i in range(batch_size):
            #flattened_predictions = predictions[i].nonzero().cpu().numpy()
            #flattened_gold_labels = gold_labels[i].nonzero().cpu().numpy()

            for j in range(len(predictions[i])):
                preds = predictions[i][j].nonzero().cpu().numpy()
                golds = gold_labels[i][j].nonzero().cpu().numpy()
                #print(preds, golds)

                for prediction in preds:
                    if prediction in golds:
                        if prediction != outside_index:
                            self._true_positives[str(prediction[0])] += 1
                    else:
                        self._false_positives[str(prediction[0])] += 1
                for gold in golds:
                    if gold not in preds:
                        self._false_negatives[str(gold[0])] += 1





        #sequence_lengths = get_lengths_from_binary_sequence_mask(mask[:,:,0])#(mask)
        # argmax_predictions = predictions.max(-1)[1]

        # @AR: It is none, ignore it
        # if prediction_map is not None:
        #     argmax_predictions = torch.gather(prediction_map, 1, argmax_predictions)
        #     gold_labels = torch.gather(prediction_map, 1, gold_labels.long())

        # argmax_predictions = argmax_predictions.float()
        # tensor([[3., 2., 2., 0., 0., 3., 2., 3., 2., 3., 2., 0., 2., 0., 3., 3., 3., 3.]]) torch.Size([1, 18])

        # # Iterate over timesteps in batch.
        # batch_size = gold_labels.size(0)
        # for i in range(batch_size):
        #     length = sequence_lengths[i]
        #     curr_seq_prediction = predictions[i,:,:]
        #     curr_seq_gold_label = gold_labels[i,:,:]
        #     # sequence_prediction = argmax_predictions[i, :]
        #     # sequence_gold_label = gold_labels[i, :]

        #     # Get the list of lists of labels
        #     predictions_list = [curr_seq_prediction]
        #     for pred in predictions_list:
        #         sequence_token_labels = to_multilabel_sequence(pred.numpy(), self.vocabulary, self.tag_namespace)

        #     if length == 0:
        #         # It is possible to call this metric with sequences which are
        #         # completely padded. These contribute nothing, so we skip these rows.
        #         continue
            
        #     # Get the predictions
        #     predicted_string_labels = []
        #     sequence_pred = sequence_token_labels[:length]
        #     for label_ids in sequence_pred:
        #         curr_string_labels = []
        #         for label_id in label_ids:
        #             curr_string_label = self._label_vocabulary[label_id]
        #             curr_string_labels.append(curr_string_label)
        #         predicted_string_labels.append(curr_string_labels)
        #     # print(predicted_string_labels)

        #     # Get the golds
        #     gold_string_labels = []
        #     sequence_gold = curr_seq_gold_label[:length].tolist()
        #     for label_indices in sequence_gold:
        #         label_ids = [label_id for label_id, value in enumerate(
        #             label_indices) if value != 0]
        #         curr_string_labels = []
        #         for label_id in label_ids:
        #             curr_string_label = self._label_vocabulary[label_id]
        #             curr_string_labels.append(curr_string_label)
        #         gold_string_labels.append(curr_string_labels)
        #     # print(gold_string_labels)

        #     assert len(predicted_string_labels) == len(gold_string_labels)

        #     # Filter outside labels from the evaluation. We assume if there is
        #     # an "O" it is the only element in the list
        #     indices_to_remove = []
        #     for i in range(len(gold_string_labels)):
        #         is_pred_outside = predicted_string_labels[i][0] == "O"
        #         is_gold_outside = gold_string_labels[i][0] == "O"
        #         if is_pred_outside and is_gold_outside:
        #             indices_to_remove.append(i)
        #     indices_to_remove.reverse()
        #     for i in indices_to_remove:
        #         predicted_string_labels.pop(i)
        #         gold_string_labels.pop(i)

        #     # Compute the TPs, FPs, and FNs
        #     for i in range(len(gold_string_labels)):
        #         pred_set = set(predicted_string_labels[i])
        #         gold_set = set(gold_string_labels[i])

        #         for pred in pred_set:
        #             if pred in gold_set:
        #                 self._true_positives[pred] += 1
        #                 gold_set.remove(pred)
        #             else:
        #                 self._false_positives[pred] += 1

        #         for gold in gold_set:
        #             self._false_negatives[gold] += 1


    # @staticmethod
    # def _handle_continued_spans(spans: List[TypedStringSpan]) -> List[TypedStringSpan]:
    #     """
    #     The official CONLL 2012 evaluation script for SRL treats continued spans (i.e spans which
    #     have a `C-` prepended to another valid tag) as part of the span that they are continuing.
    #     This is basically a massive hack to allow SRL models which produce a linear sequence of
    #     predictions to do something close to structured prediction. However, this means that to
    #     compute the metric, these continuation spans need to be merged into the span to which
    #     they refer. The way this is done is to simply consider the span for the continued argument
    #     to start at the start index of the first occurrence of the span and end at the end index
    #     of the last occurrence of the span. Handling this is important, because predicting continued
    #     spans is difficult and typically will effect overall average F1 score by ~ 2 points.

    #     Parameters
    #     ----------
    #     spans : ``List[TypedStringSpan]``, required.
    #         A list of (label, (start, end)) spans.

    #     Returns
    #     -------
    #     A ``List[TypedStringSpan]`` with continued arguments replaced with a single span.
    #     """
    #     span_set: Set[TypedStringSpan] = set(spans)
    #     continued_labels: List[str] = [label[2:] for (label, span) in span_set if label.startswith("C-")]
    #     for label in continued_labels:
    #         continued_spans = {span for span in span_set if label in span[0]}

    #         span_start = min(span[1][0] for span in continued_spans)
    #         span_end = max(span[1][1] for span in continued_spans)
    #         replacement_span: TypedStringSpan = (label, (span_start, span_end))

    #         span_set.difference_update(continued_spans)
    #         span_set.add(replacement_span)

    #     return list(span_set)

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
