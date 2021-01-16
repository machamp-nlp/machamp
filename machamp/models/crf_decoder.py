from typing import Dict, Optional, List, Any, cast

import sys
import torch
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import ConditionalRandomField
from allennlp.modules import TimeDistributed
from allennlp.modules.conditional_random_field import allowed_transitions
from allennlp.training.metrics import CategoricalAccuracy, SpanBasedF1Measure, FBetaMeasure
from overrides import overrides
from torch.nn.modules.linear import Linear


@Model.register("machamp_crf_decoder")
class MachampCrfTagger(Model):
    """
    The `CrfTagger` encodes a sequence of text with a `Seq2SeqEncoder`,
    then uses a Conditional Random Field model to predict a tag for each token in the sequence.

    Registered as a `Model` with name "crf_tagger".

    # Parameters

    vocab : `Vocabulary`, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    label_encoding : `str`, optional (default=`None`)
        Label encoding to use when calculating span f1 and constraining
        the CRF at decoding time . Valid options are "BIO", "BIOUL", "IOB1", "BMES".
        Required if `calculate_span_f1` or `constrain_crf_decoding` is true.
    include_start_end_transitions : `bool`, optional (default=`True`)
        Whether to include start and end transition parameters in the CRF.
    constrain_crf_decoding : `bool`, optional (default=`None`)
        If `True`, the CRF is constrained at decoding time to
        produce valid sequences of tags. If this is `True`, then
        `label_encoding` is required. If `None` and
        label_encoding is specified, this is set to `True`.
        If `None` and label_encoding is not specified, it defaults
        to `False`.
    calculate_span_f1 : `bool`, optional (default=`None`)
        Calculate span-level F1 metrics during training. If this is `True`, then
        `label_encoding` is required. If `None` and
        label_encoding is specified, this is set to `True`.
        If `None` and label_encoding is not specified, it defaults
        to `False`.
    dropout:  `float`, optional (default=`None`)
        Dropout probability.
    verbose_metrics : `bool`, optional (default = `False`)
        If true, metrics will be returned per label class in addition
        to the overall statistics.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        Used to initialize the model parameters.
    top_k : `int`, optional (default=`1`)
        If provided, the number of parses to return from the crf in output_dict['top_k_tags'].
        Top k parses are returned as a list of dicts, where each dictionary is of the form:
        {"tags": List, "score": float}.
        The "tags" value for the first dict in the list for each data_item will be the top
        choice, and will equal the corresponding item in output_dict['tags']
    """

    def __init__(
        self,
        task: str,
        vocab: Vocabulary,
        input_dim: int,
        loss_weight: float = 1.0,
        label_encoding: Optional[str] = 'BIO',
        include_start_end_transitions: bool = True,
        constrain_crf_decoding: bool = True,
        calculate_span_f1: bool = None,
        verbose_metrics: bool = False,
        metric: str = 'span_f1',
        dataset_embeds_dim: int = 0,
        top_k: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        self.task = task
        self.input_dim = input_dim + dataset_embeds_dim
        self.loss_weight = loss_weight
        self.num_tags = self.vocab.get_vocab_size(task)
        self.top_k = top_k
        self._verbose_metrics = verbose_metrics

        self.tag_projection_layer = TimeDistributed(Linear(input_dim, self.num_tags))

        # if  constrain_crf_decoding and calculate_span_f1 are not
        # provided, (i.e., they're None), set them to True
        # if label_encoding is provided and False if it isn't.
        if constrain_crf_decoding is None:
            constrain_crf_decoding = label_encoding is not None
        if calculate_span_f1 is None:
            calculate_span_f1 = label_encoding is not None

        self.label_encoding = label_encoding
        if constrain_crf_decoding:
            if not label_encoding:
                raise ConfigurationError(
                    "constrain_crf_decoding is True, but no label_encoding was specified."
                )
            labels = self.vocab.get_index_to_token_vocabulary(task)
            constraints = allowed_transitions(label_encoding, labels)
        else:
            constraints = None

        self.include_start_end_transitions = include_start_end_transitions
        self.crf = ConditionalRandomField(
            self.num_tags, constraints, include_start_end_transitions=include_start_end_transitions
        )
        if metric == "acc":
            self.metrics = {"acc": CategoricalAccuracy()}
        elif metric == "span_f1":
            self.metrics = {"span_f1": SpanBasedF1Measure(
                self.vocab, tag_namespace=self.task, label_encoding="BIO")}
        elif metric == "multi_span_f1":
            print(f"To use \"{metric}\", please use the \"multiseq\" decoder instead.")
            sys.exit()
        elif metric == "micro-f1":
            self.metrics = {"micro-f1": FBetaMeasure(average='micro')}
        elif metric == "macro-f1":
            self.metrics = {"macro-f1": FBetaMeasure(average='macro')}
        else:
            print(f"ERROR. Metric \"{metric}\" unrecognized. Using span-based f1 score \"span_f1\" instead.")
            self.metrics = {"span_f1": SpanBasedF1Measure(
                self.vocab, tag_namespace=self.task, label_encoding="BIO")}


    @overrides
    def forward(
        self,  # type: ignore
        embedded_text: torch.LongTensor,
        gold_labels: torch.LongTensor = None,
        mask: torch.LongTensor = None,
        metadata: List[Dict[str, Any]] = None,
        ignore_loss_on_o_tags: bool = False,
        **kwargs,  # to allow for a more general dataset reader that passes args we don't need
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
            metadata containg the original words in the sentence to be tagged under a 'words' key.
        ignore_loss_on_o_tags : `bool`, optional (default = `False`)
            If True, we compute the loss only for actual spans in `tags`, and not on `O` tokens.
            This is useful for computing gradients of the loss on a _single span_, for
            interpretation / attacking.

        # Returns

        An output dictionary consisting of:

        logits : `torch.FloatTensor`
            The logits that are the output of the `tag_projection_layer`
        mask : `torch.BoolTensor`
            The text field mask for the input tokens
        tags : `List[List[int]]`
            The predicted tags using the Viterbi algorithm.
        loss : `torch.FloatTensor`, optional
            A scalar loss to be optimised. Only computed if gold label `tags` are provided.
        """

        logits = self.tag_projection_layer(embedded_text)
        best_paths = self.crf.viterbi_tags(logits, mask, top_k=self.top_k)

        # Just get the top tags and ignore the scores.
        predicted_tags = cast(List[List[int]], [x[0][0] for x in best_paths])

        output = {}

        #if self.top_k > 1:
        #    output["top_k_tags"] = best_paths

        if gold_labels is not None:
            if ignore_loss_on_o_tags:
                o_tag_index = self.vocab.get_token_index("O", namespace=task)
                crf_mask = mask & (gold_labels != o_tag_index)
            else:
                crf_mask = mask
            # Add negative log-likelihood as loss
            log_likelihood = self.crf(logits, gold_labels, crf_mask)

            output["loss"] = -log_likelihood * self.loss_weight

            # Represent viterbi tags as "class probabilities" that we can
            # feed into the metrics
            class_probabilities = logits * 0.0
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1

            for metric in self.metrics.values():
                metric(class_probabilities, gold_labels, mask)

        output['class_probabilities'] = predicted_tags
        return output

    @overrides
    def make_output_human_readable(
        self, predictions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        `output_dict["tags"]` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """
        
        def decode_tags(tags):
            return [
                self.vocab.get_token_from_index(tag, namespace=self.task) for tag in tags
            ]

        def decode_top_k_tags(top_k_tags):
            return [
                {"tags": decode_tags(scored_path[0]), "score": scored_path[1]}
                for scored_path in top_k_tags
            ]
        #output_dict["tags"] = [decode_tags(t) for t in output_dict["tags"]]

        #if "top_k_tags" in output_dict:
        #    output_dict["top_k_tags"] = [decode_top_k_tags(t) for t in output_dict["top_k_tags"]]
        return [decode_tags(t) for t in predictions]


    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        main_metrics = {
            f".run/{self.task}/{metric_name}": metric.get_metric(reset)
            for metric_name, metric in self.metrics.items()
        }
        return {**main_metrics}

