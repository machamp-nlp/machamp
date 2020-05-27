from typing import Dict, Optional, List, Any, cast

from overrides import overrides
import numpy
import torch
import logging
from torch.nn.modules.linear import Linear

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary#TextFieldTensors, Vocabulary
from allennlp.modules import Seq2SeqEncoder, TimeDistributed, TextFieldEmbedder
from allennlp.modules import FeedForward#ConditionalRandomField, FeedForward
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
import allennlp.nn.util as util
from allennlp.training.metrics import CategoricalAccuracy
from allennlp.training.metrics import FBetaMeasure
from machamp.metrics.span_based_f1_measure import SpanBasedF1Measure

from machamp.modules import masked_conditional_random_field
from machamp.modules.masked_conditional_random_field import allowed_transitions

logger = logging.getLogger(__name__)

# from: https://github.com/allenai/allennlp/blob/master/allennlp/models/crf_tagger.py
# Last check: 28/01/2020


@Model.register("masked_crf_decoder")
class MaskedCrfDecoder(Model):
    """
    The `CrfTagger` encodes a sequence of text with a `Seq2SeqEncoder`,
    then uses a Conditional Random Field model to predict a tag for each token in the sequence.
    # Parameters
    vocab : `Vocabulary`, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : `TextFieldEmbedder`, required
        Used to embed the tokens `TextField` we get as input to the model.
    encoder : `Seq2SeqEncoder`
        The encoder that we will use in between embedding tokens and predicting output tags.
    label_namespace : `str`, optional (default=`labels`)
        This is needed to compute the SpanBasedF1Measure metric.
        Unless you did something unusual, the default value should be what you want.
    feedforward : `FeedForward`, optional, (default = None).
        An optional feedforward layer to apply after the encoder.
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
    verbose_metrics : `bool`, optional (default = False)
        If true, metrics will be returned per label class in addition
        to the overall statistics.
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        Used to initialize the model parameters.
    regularizer : `RegularizerApplicator`, optional (default=`None`)
        If provided, will be used to calculate the regularization penalty during training.
    top_k : `int`, optional (default=`1`)
        If provided, the number of parses to return from the crf in output_dict['top_k_tags'].
        Top k parses are returned as a list of dicts, where each dictionary is of the form:
        {"tags": List, "score": float}.
        The "tags" value for the first dict in the list for each data_item will be the top
        choice, and will equal the corresponding item in output_dict['tags']
    """

    def __init__(
        self,
        vocab: Vocabulary,
        #text_field_embedder: TextFieldEmbedder,
        task: str,

        encoder: Seq2SeqEncoder,

        prev_task: str,
        prev_task_embed_dim: int = None,
        task_types: List[str] = None,
        adaptive: bool = False,
        tasks: List[str] = None,
        loss_weight: float = 0.0,
        label_smoothing: float = 0.0,
        metric: str = "span_f1",
        features: List[str] = None,
        token_label_constraint: str = None,
        orphan_entity_as_label: bool = False,

        #label_namespace: str = "bio-labels",
        feedforward: Optional[FeedForward] = None,
        label_encoding: Optional[str] = None,
        include_start_end_transitions: bool = True,
        constrain_crf_decoding: bool = None,
        calculate_span_f1: bool = None,
        dropout: Optional[float] = None,
        verbose_metrics: bool = False,
        initializer: InitializerApplicator = InitializerApplicator(),
        regularizer: Optional[RegularizerApplicator] = None,
        top_k: int = 1,
    ) -> None:
        super().__init__(vocab, regularizer)

        #self.label_namespace = task#label_namespace
        #self.text_field_embedder = text_field_embedder
        self.num_tags = self.vocab.get_vocab_size(task) #label_namespace)
        self.encoder = encoder
        self.top_k = top_k
        self._verbose_metrics = verbose_metrics
        if dropout:
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None
        self._feedforward = feedforward

        if feedforward is not None:
            output_dim = feedforward.get_output_dim()
        else:
            output_dim = self.encoder.get_output_dim()
        self.tag_projection_layer = TimeDistributed(Linear(output_dim, self.num_tags))

        self.task = task
        self.metric = metric
        self.features = features if features else []

        # Set the token-level constraint according to the available ones
        supported_constraints = ["entity"]
        if token_label_constraint in supported_constraints:
            self.token_label_constraint = token_label_constraint
        else:
            self.token_label_constraint = None
        self.orphan_entity_as_label = orphan_entity_as_label

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
            labels = self.vocab.get_index_to_token_vocabulary(self.task) #label_namespace)
            constraints = allowed_transitions(label_encoding, labels)
        else:
            constraints = None

        self.include_start_end_transitions = include_start_end_transitions
        self.crf = masked_conditional_random_field.ConditionalRandomField(
            self.num_tags, self.task, constraints, include_start_end_transitions=include_start_end_transitions,
            token_label_constraint=self.token_label_constraint, 
            labels=self.vocab.get_token_to_index_vocabulary(task),
            orphan_entity_as_label=self.orphan_entity_as_label,
        )

        # @AR: Choose the metric to use for the evaluation (from the defined
        # "metric" value of the task). If not specified, default to accuracy.
        if self.metric == "acc":
            self.metrics = {"acc": CategoricalAccuracy()}
        elif self.metric == "micro-f1":
            self.metrics = {"micro-f1": FBetaMeasure(average='micro')}
        elif self.metric == "macro-f1":
            self.metrics = {"macro-f1": FBetaMeasure(average='macro')}
        elif self.metric == "span_f1":
            self.metrics = {"span_f1": SpanBasedF1Measure(
                self.vocab, tag_namespace=self.task, label_encoding="BIO")}
        else:
            logger.warning(f"ERROR. Metric: {self.metric} unrecognized. Using accuracy instead.")
            self.metrics = {"acc": CategoricalAccuracy()}
        # self.metrics = {
        #     "acc": CategoricalAccuracy(),
        #     "accuracy3": CategoricalAccuracy(top_k=3),
        # }
        # self.calculate_span_f1 = calculate_span_f1
        # if calculate_span_f1:
        #     if not label_encoding:
        #         raise ConfigurationError(
        #             "calculate_span_f1 is True, but no label_encoding was specified."
        #         )
        #     self._f1_metric = SpanBasedF1Measure(
        #         vocab, tag_namespace=label_namespace, label_encoding=label_encoding
        #     )

        # check_dimensions_match(
        #     text_field_embedder.get_output_dim(),
        #     encoder.get_input_dim(),
        #     "text field embedding dim",
        #     "encoder input dim",
        # )
        if feedforward is not None:
            check_dimensions_match(
                encoder.get_output_dim(),
                feedforward.get_input_dim(),
                "encoder output dim",
                "feedforward input dim",
            )
        initializer(self)

    @overrides
    def forward(
        self,  # type: ignore
        #tokens: TextFieldTensors,
        encoded_text: torch.FloatTensor,
        mask: torch.LongTensor,
        #tags: torch.LongTensor = None, # @AR; must be gold_tags: Dict[str, torch.LongTensor],
        gold_tags: Dict[str, torch.LongTensor],

        prev_task_classes: torch.LongTensor = None,

        metadata: List[Dict[str, Any]] = None,
        **kwargs # @AR: not necessary
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
        metadata : `List[Dict[str, Any]]`, optional, (default = None)
            metadata containg the original words in the sentence to be tagged under a 'words' key.
        # Returns
        An output dictionary consisting of:
        logits : `torch.FloatTensor`
            The logits that are the output of the `tag_projection_layer`
        mask : `torch.LongTensor`
            The text field mask for the input tokens
        tags : `List[List[int]]`
            The predicted tags using the Viterbi algorithm.
        loss : `torch.FloatTensor`, optional
            A scalar loss to be optimised. Only computed if gold label `tags` are provided.
        """
        #embedded_text_input = self.text_field_embedder(tokens)
        #mask = util.get_text_field_mask(tokens)

        #if self.dropout:
        #    embedded_text_input = self.dropout(embedded_text_input)

        #encoded_text = self.encoder(embedded_text_input, mask)

        #if self.dropout:
        #    encoded_text = self.dropout(encoded_text)

        batch_size = list(encoded_text.shape)[0]
        entity_mask = []
        for batch in range(batch_size):
            batch_mask = []
            for tok_i in range(1, len(metadata[batch]["fullData"])):
                if metadata[batch]["fullData"][tok_i][3].split("]")[1] == "-":
                    batch_mask.append(0)
                else:
                    batch_mask.append(1)
            entity_mask.append(batch_mask)

        tags = gold_tags[self.task]

        if self._feedforward is not None:
            encoded_text = self._feedforward(encoded_text)

        logits = self.tag_projection_layer(encoded_text)

        # @AR: logits:      batch_size * tokens * labels
        # @AR: mask:        batch_size * tokens
        best_paths = self.crf.viterbi_tags(logits, mask, entity_mask, top_k=self.top_k)

        # Just get the top tags and ignore the scores.
        predicted_tags = cast(List[List[int]], [x[0][0] for x in best_paths])

        output = {"logits": logits, "mask": mask, "tags": predicted_tags}

        if self.top_k > 1:
            output["top_k_tags"] = best_paths

        if tags is not None:
            # Add negative log-likelihood as loss
            log_likelihood = self.crf(logits, tags, mask, entity_mask)

            output["loss"] = -log_likelihood

            # Represent viterbi tags as "class probabilities" that we can
            # feed into the metrics
            # @AR: class_probabilities: batch_size * tokens * labels
            class_probabilities = logits * 0.0
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1

            for metric in self.metrics.values():
                metric(class_probabilities, tags, mask.float())
            #if self.calculate_span_f1:
            #    self._f1_metric(class_probabilities, tags, mask.float())
        if metadata is not None:
            output["words"] = [x["words"] for x in metadata]

        output["logits"] = logits
        output["class_probabilities"] = class_probabilities

        return output

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Converts the tag ids to the actual tags.
        `output_dict["tags"]` is a list of lists of tag_ids,
        so we use an ugly nested list comprehension.
        """

        def decode_tags(tags):
            return [
                #self.vocab.get_token_from_index(tag, namespace=self.label_namespace) for tag in tags
                self.vocab.get_token_from_index(tag, namespace=self.task) for tag in tags
            ]

        def decode_top_k_tags(top_k_tags):
            return [
                {"tags": decode_tags(scored_path[0]), "score": scored_path[1]}
                for scored_path in top_k_tags
            ]

        all_words = output_dict["words"]

        all_predictions = output_dict["class_probabilities"][self.task].cpu().data.numpy()
        if all_predictions.ndim == 3:
            predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        for predictions, words in zip(predictions_list, all_words):
            argmax_indices = numpy.argmax(predictions, axis=-1)
            tags = [self.vocab.get_token_from_index(x, namespace=self.task)
                    for x in argmax_indices]
            all_tags.append(tags)
        output_dict[self.task] = all_tags

        # output_dict["tags"] = [decode_tags(t) for t in output_dict["tags"]]

        # if "top_k_tags" in output_dict:
        #     output_dict["top_k_tags"] = [decode_top_k_tags(t) for t in output_dict["top_k_tags"]]

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        main_metrics = {
            f".run/{self.task}/{metric_name}": metric.get_metric(reset)
            for metric_name, metric in self.metrics.items()
        }

        features_metrics = {
            f"_run/{self.task}/{feature}/{metric_name}": metric.get_metric(reset)
            for feature in self.features
            for metric_name, metric in self.features_metrics[feature].items()
        }

        return {**main_metrics, **features_metrics}

    # @overrides
    # def get_metrics(self, reset: bool = False) -> Dict[str, float]:
    #     metrics_to_return = {
    #         metric_name: metric.get_metric(reset) for metric_name, metric in self.metrics.items()
    #     }

    #     if self.calculate_span_f1:
    #         f1_dict = self._f1_metric.get_metric(reset=reset)
    #         if self._verbose_metrics:
    #             metrics_to_return.update(f1_dict)
    #         else:
    #             metrics_to_return.update({x: y for x, y in f1_dict.items() if "overall" in x})
    #     return metrics_to_return
