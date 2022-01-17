from typing import Dict, Optional, List, Union

import sys
import numpy
import torch
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import TimeDistributed
from overrides import overrides
from torch.nn.modules.linear import Linear
from allennlp.training.metrics import CategoricalAccuracy, FBetaMeasure, FBetaMultiLabelMeasure

from machamp.metrics.multi_span_based_f1_measure import MultiSpanBasedF1Measure
from machamp.metrics.multi_accuracy import MultiAccuracy


@Model.register("machamp_multiseq_decoder")
class MachampMultiTagger(Model):
    """
    This `SimpleTagger` simply encodes a sequence of text with a stacked `Seq2SeqEncoder`, then
    predicts a tag for each token in the sequence.

    Registered as a `Model` with name "simple_tagger".

    # Parameters

    vocab : `Vocabulary`, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : `TextFieldEmbedder`, required
        Used to embed the `tokens` `TextField` we get as input to the model.
    encoder : `Seq2SeqEncoder`
        The encoder (with its own internal stacking) that we will use in between embedding tokens
        and predicting output tags.
    calculate_span_f1 : `bool`, optional (default=`None`)
        Calculate span-level F1 metrics during training. If this is `True`, then
        `label_encoding` is required. If `None` and
        label_encoding is specified, this is set to `True`.
        If `None` and label_encoding is not specified, it defaults
        to `False`.
    label_encoding : `str`, optional (default=`None`)
        Label encoding to use when calculating span f1.
        Valid options are "BIO", "BIOUL", "IOB1", "BMES".
        Required if `calculate_span_f1` is true.
    verbose_metrics : `bool`, optional (default = `False`)
        If true, metrics will be returned per label class in addition
        to the overall statistics.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        Used to initialize the model parameters.
    """

    def __init__(
        self,
        task: str,
        vocab: Vocabulary,
        input_dim: int,
        loss_weight: float = 1.0,
        metric: str = 'acc',
        label_encoding: Optional[str] = None,
        verbose_metrics: bool = False,
        threshold: float = 0.5,
        dec_dataset_embeds_dim: int = 0,
        max_heads: int = 0,
        #focal_alpha: float = None) -> None:
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)
        self.task = task
        self.vocab = vocab
        self.input_dim = input_dim + dec_dataset_embeds_dim
        self.loss_weight = loss_weight
        self.num_classes = self.vocab.get_vocab_size(task)
        self._verbose_metrics = verbose_metrics
        self.tag_projection_layer = TimeDistributed(
            Linear(input_dim, self.num_classes)
        )
        self.threshold = threshold
        self.max_heads = max_heads

        #if metric == "f1":TODO
        #    self.metrics = {"acc": FBetaMultiLabelMeasure()}
        if metric == "acc":
            self.metrics = {"acc": MultiAccuracy()}
        elif metric == "span_f1":
            print(f"To use \"{metric}\", please use the \"seq_bio\" decoder instead.")
            sys.exit()
        elif metric == "multi_span_f1":
            self.metrics = {"multi_span_f1": MultiSpanBasedF1Measure(
                self.vocab, tag_namespace=self.task, label_encoding="BIO", 
                threshold=self.threshold, max_heads=self.max_heads)}
        else:
            print(f"ERROR. Metric \"{metric}\" not supported for task-type multiseq. Use multi span-based f1 score \"multi_span_f1\" or accuracy instead.")
            exit(1)

    @overrides
    def forward(
        self,  # type: ignore
        embedded_text: torch.LongTensor,
        gold_labels: torch.LongTensor = None,
        mask: torch.LongTensor = None,
        ignore_loss_on_o_tags: bool = False,
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters

        tokens : `TextFieldTensors`, required
            The output of `TextField.as_array()`, which should typically be passed directly to a
            `TextFieldEmbedder`. This output is a dictionary mapping keys to `TokenIndexer`
            tensors.  At its most basic, using a `SingleIdTokenIndexer` this is : `{"tokens":
            Tensor(batch_size, num_tokens)}`. This dictionary will have the same keys as were used
            for the `TokenIndexers` when you created the `TextField` representing your
            sequence.  The dictionary is designed to be passed directly to a `TextFieldEmbedder`,
            which knows how to combine different word representations into a single vector per
            token in your input.
        tags : `torch.LongTensor`, optional (default = `None`)
            A torch tensor representing the sequence of integer gold class labels of shape
            `(batch_size, num_tokens)`.
        metadata : `List[Dict[str, Any]]`, optional, (default = `None`)
            metadata containing the original words in the sentence to be tagged under a 'words' key.
        ignore_loss_on_o_tags : `bool`, optional (default = `False`)
            If True, we compute the loss only for actual spans in `tags`, and not on `O` tokens.
            This is useful for computing gradients of the loss on a _single span_, for
            interpretation / attacking.

        # Returns

        An output dictionary consisting of:
            - `logits` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_tokens, tag_vocab_size)` representing
                unnormalised log probabilities of the tag classes.
            - `class_probabilities` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_tokens, tag_vocab_size)` representing
                a distribution of the tag classes per word.
            - `loss` (`torch.FloatTensor`, optional) :
                A scalar loss to be optimised.

        """
        batch_size, sequence_length, _ = embedded_text.shape

        logits = self.tag_projection_layer(embedded_text)
        reshaped_log_probs = logits.view(-1, self.num_classes)
        #class_probabilities = F.softmax(reshaped_log_probs, dim=-1).view(
        #    [batch_size, sequence_length, self.num_classes]
        #)
        class_probabilities = torch.sigmoid(logits)

        output_dict = {"logits": logits, "class_probabilities": class_probabilities}

        if gold_labels is not None:
            if ignore_loss_on_o_tags:
                o_tag_index = self.vocab.get_token_index("O", namespace=self.task)
                tag_mask = mask & (gold_labels != o_tag_index)
            else:
                tag_mask = mask
            output_dict["loss"] = self.multi_class_cross_entropy_loss(logits, gold_labels, tag_mask) * self.loss_weight
            for metric in self.metrics:
                if metric == 'multi_span_f1':
                    self.metrics[metric](class_probabilities, gold_labels, mask)
                else:
                    self.metrics[metric](class_probabilities > self.threshold, gold_labels)
        return output_dict

    def multi_class_cross_entropy_loss(self,
                                       logits: torch.FloatTensor,
                                       targets: torch.LongTensor,
                                       weights: torch.FloatTensor,
                                       average: str = "batch",
                                       gamma: float = None,
                                       alpha: Union[float, List[float], torch.FloatTensor] = None
                                      ) -> torch.FloatTensor:
        """
        Computes the cross entropy loss of a sequence, weighted with respect to
        some user provided weights. Note that the weighting here is not the same as
        in the :func:`torch.nn.CrossEntropyLoss()` criterion, which is weighting
        classes; here we are weighting the loss contribution from particular elements
        in the sequence. This allows loss computations for models which use padding.

        Parameters
        ----------
        logits : ``torch.FloatTensor``, required.
            A ``torch.FloatTensor`` of size (batch_size, sequence_length, num_classes)
            which contains the unnormalized probability for each class.
        targets : ``torch.LongTensor``, required.
            A ``torch.LongTensor`` of size (batch, sequence_length) which contains the
            index of the true class for each corresponding step.
        weights : ``torch.FloatTensor``, required.
            A ``torch.FloatTensor`` of size (batch, sequence_length)
        average: str, optional (default = "batch")
            If "batch", average the loss across the batches. If "token", average
            the loss across each item in the input. If ``None``, return a vector
            of losses per batch element.
        gamma : ``float``, optional (default = None)
            Focal loss[*] focusing parameter ``gamma`` to reduces the relative loss for
            well-classified examples and put more focus on hard. The greater value
            ``gamma`` is, the more focus on hard examples.
        alpha : ``float`` or ``List[float]``, optional (default = None)
            Focal loss[*] weighting factor ``alpha`` to balance between classes. Can be
            used independently with ``gamma``. If a single ``float`` is provided, it
            is assumed binary case using ``alpha`` and ``1 - alpha`` for positive and
            negative respectively. If a list of ``float`` is provided, with the same
            length as the number of classes, the weights will match the classes.
            [*] T. Lin, P. Goyal, R. Girshick, K. He and P. Dollár, "Focal Loss for
            Dense Object Detection," 2017 IEEE International Conference on Computer
            Vision (ICCV), Venice, 2017, pp. 2999-3007.

        Returns
        -------
        A torch.FloatTensor representing the cross entropy loss.
        If ``average=="batch"`` or ``average=="token"``, the returned loss is a scalar.
        If ``average is None``, the returned loss is a vector of shape (batch_size,).

        """
        if average not in {None, "token", "batch"}:
            raise ValueError("Got average f{average}, expected one of "
                             "None, 'token', or 'batch'")



        # make sure weights are float
        # weights = weights.float()

        # Compute the mask before computing the loss
        # Transform the mask that is at the sentence level (#Size: n_batches x padded_document_length)
        # to a suitable format for the relation labels level
        #mask (2x3)
        padded_document_length = weights.size(1) # prendi la seconda dimensione (3)
        weights = weights.float()  # Size: n_batches x padded_document_length (2x3)



        # @AR: Make weights be of the right shape (i.e., extend a dimension to NUM_CLASSES)
        NUM_CLASSES = logits.size(-1)
        #weights = weights.unsqueeze_(-1)
        #weights = weights.expand(weights.shape[0], weights.shape[1], NUM_CLASSES)
        #weights = weights.unsqueeze(2).expand(weights.shape[0], weights.shape[1], NUM_CLASSES)

        # [e.view(padded_document_length, 1) * e for e in mask] ([3x3, 3x3])
        #squared_mask = torch.stack([e.view(padded_document_length, 1) * e for e in mask], dim=0) (2x3x3)
        #squared_mask = squared_mask.unsqueeze(-1).repeat(
        weights = weights.unsqueeze(-1).repeat(
            #1, 1, 1, self._n_classes
            1, 1, logits.size(-1)
        )  # Size: n_batches x padded_document_length x padded_document_length x n_classes (2x3x3x5)


        # sum all dim except batch
        non_batch_dims = tuple(range(1, len(weights.shape)))

        # shape : (batch_size,)
        weights_batch_sum = weights.sum(dim=non_batch_dims)
        weights_batch_sum2 = weights.sum(dim=(1,))[:,0]

        # shape : (batch * sequence_length, num_classes)
        # logits_flat = logits.view(-1, logits.size(-1))

        # @AR: Use log_sigmoid instead of log_softmax
        # log_probs_flat = torch.nn.functional.logsigmoid(logits_flat)
        # shape : (batch * sequence_length, num_classes)
        # log_probs_flat = torch.nn.functional.log_softmax(logits_flat, dim=-1)

        # @AR: Make the target handle NUM_CLASSES instead of one-best
        # shape : (batch * max_len, NUM_CLASSES)
        # targets_flat = targets.view(-1, NUM_CLASSES)
        # shape : (batch * max_len, 1)
        # targets_flat = targets.view(-1, 1).long()


        # The scores (and gold labels) are flattened before using
        # the binary cross entropy loss.
        # We thus transform
        flat_size = logits.size()
        logits = logits * weights  # Size: n_batches x padded_document_length x padded_document_length x n_classes
        logits_flat = logits.view(
            flat_size[0], flat_size[1] * logits.size(-1)
        #    flat_size[0], flat_size[1], flat_size[2] * self._n_classes
        )  # Size: n_batches x padded_document_length x (padded_document_length x n_classes)
        targets = targets * weights  # Size: n_batches x padded_document_length x padded_document_length x n_classes
        targets_flat = targets.view(
            flat_size[0], flat_size[1] * logits.size(-1)
        #    flat_size[0], flat_size[1], flat_size[2] * self._n_classes
        )  # Size: n_batches x padded_document_length x (padded_document_length x n_classes)



        # focal loss coefficient
        # if gamma:
        #     # shape : (batch * sequence_length, num_classes)
        #     probs_flat = log_probs_flat.exp()
        #     # shape : (batch * sequence_length,)
        #     probs_flat = torch.gather(probs_flat, dim=1, index=targets_flat)
        #     # shape : (batch * sequence_length,)
        #     focal_factor = (1. - probs_flat) ** gamma
        #     # shape : (batch, sequence_length)
        #     focal_factor = focal_factor.view(*targets.size())
        #     weights = weights * focal_factor

        if alpha is not None:
            # shape : () / (num_classes,)
            if isinstance(alpha, (float, int)):
                # pylint: disable=not-callable
                # shape : (2,)
                alpha_factor = torch.tensor([1. - float(alpha), float(alpha)],
                                            dtype=weights.dtype, device=weights.device)
                # pylint: enable=not-callable
            elif isinstance(alpha, (list, numpy.ndarray, torch.Tensor)):
                # pylint: disable=not-callable
                # shape : (c,)
                alpha_factor = torch.tensor(alpha, dtype=weights.dtype, device=weights.device)
                # pylint: enable=not-callable
                if not alpha_factor.size():
                    # shape : (1,)
                    alpha_factor = alpha_factor.view(1)
                    # shape : (2,)
                    alpha_factor = torch.cat([1 - alpha_factor, alpha_factor])
            else:
                raise TypeError(('alpha must be float, list of float, or torch.FloatTensor, '
                                 '{} provided.').format(type(alpha)))
            # shape : (batch, max_len)
            #alpha_factor = torch.gather(alpha_factor, dim=0, index=targets_flat.view(-1)).view(*targets.size())
            #weights = weights * alpha_factor

        # Contribution to the negative log likelihood only comes from the exact indices
        # of the targets, as the target distributions are one-hot. Here we use torch.gather
        # to extract the indices of the num_classes dimension which contribute to the loss.
        # shape : (batch * sequence_length, 1)
        # negative_log_likelihood_flat = - torch.gather(log_probs_flat, dim=1, index=targets_flat)
        # negative_log_likelihood_flat = - log_probs_flat
        negative_log_likelihood_ = torch.nn.functional.binary_cross_entropy_with_logits(logits_flat, targets_flat, reduction='none') #self._loss3(logits_new, targets_new)
        # shape : (batch, sequence_length)
        # negative_log_likelihood = negative_log_likelihood_.view(*targets.size())
        # negative_log_likelihood = negative_log_likelihood_flat.view(*targets.size())
        # shape : (batch, sequence_length)
        #negative_log_likelihood = negative_log_likelihood * weights

        if gamma:
            # shape : (batch * sequence_length, num_classes)
            # probs_flat = log_probs_flat.exp()
            probs_flat = negative_log_likelihood_.exp()
            # shape : (batch * sequence_length,)
            # probs_flat = torch.gather(probs_flat, dim=1, index=targets_flat)
            # shape : (batch * sequence_length,)
            focal_factor = (1. - probs_flat) ** gamma
            # shape : (batch, sequence_length)
            focal_factor = focal_factor.view(*targets.size())
            weights = weights * focal_factor

        if alpha is not None:
            # shape : (batch, max_len)
            alpha_factor = torch.gather(alpha_factor, dim=0, index=targets_flat.long().view(-1)).view(*targets.size())
            weights = weights * alpha_factor

        negative_log_likelihood = negative_log_likelihood_.view(*targets.size())
        negative_log_likelihood = negative_log_likelihood * weights


        if average == "batch":
            # shape : (batch_size,)
            per_token_loss = negative_log_likelihood.sum((2,)) / NUM_CLASSES

            #per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (weights_batch_sum + 1e-13)
            per_batch_loss = per_token_loss.sum((1,)) / (weights_batch_sum2 + 1e-13)

            num_non_empty_sequences = ((weights_batch_sum2 > 0).float().sum() + 1e-13)

            # amplify it to see something
            # inspired by https://github.com/huggingface/hmtl/blob/master/hmtl/models/relation_extraction.py#L131
            return (per_batch_loss.sum() / num_non_empty_sequences) * 100 
        elif average == "token":
            return negative_log_likelihood.sum() / (weights_batch_sum.sum() + 1e-13)
        else:
            # shape : (batch_size,)
            per_batch_loss = negative_log_likelihood.sum(non_batch_dims) / (weights_batch_sum + 1e-13)
            return per_batch_loss


    @overrides
    def make_output_human_readable(
        self, predictions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Does a simple position-wise argmax over each token, converts indices to string labels, and
        adds a `"tags"` key to the dictionary with the result.
        """
        all_predictions = predictions.cpu().data.numpy()
        if all_predictions.ndim == 3:
            predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        for predictions in predictions_list:
            outside_index = self.vocab.get_token_index("O", namespace=self.task)

            # @AR: Get the thresholded matrix and prepare the prediction sequence
            pred_over_thresh = (predictions >= self.threshold) * predictions
            #print(pred_over_thresh)
            sequence_token_labels = []
            maxxx = numpy.argmax(predictions, axis=-1).tolist()

            # @AR: For each label set, check if to apply argmax or sigmoid thresh
            j=0
            for pred in pred_over_thresh:
                num_pred_over_thresh = numpy.count_nonzero(pred)
                pred_idx_list = None
                if (num_pred_over_thresh == 0) or (num_pred_over_thresh == 1):
                    pred_idx_list = [maxxx[j]]

                else:
                    try:
                        if pred_idx_list != None:
                            outside_position = pred_idx_list.index(outside_index)
                        else:
                            outside_position = -1
                    except ValueError:
                        outside_position = -1
                    # get ranked list
                    tuples = [[score, idx] for idx, score in enumerate(pred) if score > self.threshold and idx != outside_position]
                    # check for max_heads
                    if self.max_heads != 0 and len(tuples) > self.max_heads:
                        tuples = tuples[:self.max_heads]
                    if len(tuples) == 0:
                        tuples = [1.0, outside_position]
                    pred_idx_list = [x[1] for x in tuples]
                    

                sequence_token_labels.append(pred_idx_list)
                j += 1

            # @AR: Create the list of tags to append for the output
            tags = []
            for token_labels in sequence_token_labels:
                curr_labels = []
                for token_label in token_labels:
                    curr_labels.append(
                        self.vocab.get_token_from_index(token_label, namespace=self.task))
                tags.append(curr_labels)

            all_tags.append(tags)
        return all_tags


    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        main_metrics = {
            f".run/{self.task}/{metric_name}": metric.get_metric(reset)
            for metric_name, metric in self.metrics.items()
        }
        return {**main_metrics}

