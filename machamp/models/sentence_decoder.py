import logging
from collections import Counter
from typing import Dict, Union, Optional

import sys
import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides

from machamp.metrics.fbeta_measure import FBetaMeasure

logger = logging.getLogger(__name__)

@Model.register("machamp_sentence_decoder")
class MachampClassifier(Model):
    """
    This `Model` implements a basic text classifier. After embedding the text into
    a text field, we will optionally encode the embeddings with a `Seq2SeqEncoder`. The
    resulting sequence is pooled using a `Seq2VecEncoder` and then passed to
    a linear classification layer, which projects into the label space. If a
    `Seq2SeqEncoder` is not provided, we will pass the embedded text directly to the
    `Seq2VecEncoder`.
    Registered as a `Model` with name "basic_classifier".
    # Parameters
    vocab : `Vocabulary`
    text_field_embedder : `TextFieldEmbedder`
        Used to embed the input text into a `TextField`
    initializer : `InitializerApplicator`, optional (default=`InitializerApplicator()`)
        If provided, will be used to initialize the model parameters.
    """

    def __init__(
        self,
        task: str,
        vocab: Vocabulary,
        input_dim: int,
        loss_weight: float=1.0,
        class_weights: Optional[Union[str, Dict[str, float]]] = None,
        dec_dataset_embeds_dim: int = 0,
        metric: str = "acc",
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)

        self.task = task
        self.vocab = vocab
        self.input_dim = input_dim + dec_dataset_embeds_dim
        self.loss_weight = loss_weight
        self.metric = metric
        self.num_labels = self.vocab.get_vocab_size(namespace=task)

        self._classifier_input_dim = self.input_dim

        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self.num_labels)
        
        self.class_weights = class_weights

        if self.metric == "acc":
            self.metrics = {"acc": CategoricalAccuracy()}
        elif self.metric == "span_f1":
            print(f"To use \"{metric}\", please use the \"seq_bio\" decoder instead.")
            sys.exit()
        elif self.metric == "multi_span_f1":
            print(f"To use \"{metric}\", please use the \"multiseq\" decoder instead.")
            sys.exit()
        elif self.metric == "micro-f1":
            self.metrics = {"micro-f1": FBetaMeasure(average='micro')}
        elif self.metric == "macro-f1":
            self.metrics = {"macro-f1": FBetaMeasure(average='macro')}
        else:
            print(f"ERROR. Metric \"{metric}\" unrecognized. Using accuracy \"acc\" instead.")
            self.metrics = {"acc": CategoricalAccuracy()}


    def forward(  # type: ignore
        self, 
        embedded_text: TextFieldTensors, 
        gold_labels: torch.LongTensor = [],
        label_counts: Dict[str, float] = None
    ) -> Dict[str, torch.Tensor]:

        """
        # Parameters
        tokens : `TextFieldTensors`
            From a `TextField`
        label : `torch.IntTensor`, optional (default = `None`)
            From a `LabelField`
        # Returns
        An output dictionary consisting of:
            - `logits` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_labels)` representing
                unnormalized log probabilities of the label.
            - `probs` (`torch.FloatTensor`) :
                A tensor of shape `(batch_size, num_labels)` representing
                probabilities of the label.
            - `loss` : (`torch.FloatTensor`, optional) :
                A scalar loss to be optimised.
        """
        logits = self._classification_layer(embedded_text)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        # Compute class weights for cross entropy loss if the class weights parameter is set (executed only at the first forward call)
        if self.class_weights is not None:
            # If they are explicitly defined in the config file, use these
            if type(self.class_weights) is dict:
                if len(self.class_weights.values()) > self.num_labels:
                    logger.error(f"ERROR. Class weights must be {self.num_labels}, but {len(self.class_weights.values())} are defined.")
                
                weights: List[float] = [0.0] * self.num_labels
                for label, weight in self.class_weights.items():
                    label_idx = self.vocab.get_token_index(label, namespace=self.task)
                    weights[label_idx] = weight
                self.class_weights = torch.FloatTensor(weights).cuda()

            # If they are set to True, compute them automatically
            elif (self.class_weights == "balanced"):
                num_samples, num_labels = sum(label_counts.values()), len(label_counts.values())

                weights: List[float] = [0.0] * self.num_labels
                for label, label_count in label_counts.items():
                    weight = num_samples / float(num_labels * label_count)
                    label_idx = self.vocab.get_token_index(label, namespace=self.task)
                    weights[label_idx] = weight
                self.class_weights = torch.FloatTensor(weights).cuda()

            # Class weights are already initialized
            else:
                pass

        output_dict = {"logits": logits, "class_probabilities": probs}

        if gold_labels != None:
            if self.class_weights is not None:
                loss = torch.nn.functional.cross_entropy(logits, gold_labels.long().view(-1), self.class_weights)
            else:
                loss = torch.nn.functional.cross_entropy(logits, gold_labels.long().view(-1))
            output_dict['loss'] = loss * self.loss_weight
            for metric in self.metrics.values():
                metric(logits, gold_labels)

        return output_dict

    @overrides
    def make_output_human_readable(
        self, predictions: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the probabilities, converts index to string label, and
        add `"label"` key to the dictionary with the result.
        """
        if predictions.dim() == 2:
            predictions_list = [predictions[i] for i in range(predictions.shape[0])]
        else:
            predictions_list = [predictions]
        classes = []
        for prediction in predictions_list:
            label_idx = prediction.argmax(dim=-1).item()
            label_str = self.vocab.get_index_to_token_vocabulary(self.task).get(
                label_idx, str(label_idx)
            )
            classes.append(label_str)
        return classes

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        main_metrics = {}
        for metric_name, metric in self.metrics.items():
            if metric_name.endswith('f1'):
                if metric._true_positive_sum == None:
                    main_metrics[f".run/{self.task}/{metric_name}"] = {'precision':0.0, 'recall': 0.0, 'fscore': 0.0}
                else:
                    main_metrics[f".run/{self.task}/{metric_name}"] = metric.get_metric(reset)
            else:
                main_metrics[f".run/{self.task}/{metric_name}"] = metric.get_metric(reset)
        return {**main_metrics}

