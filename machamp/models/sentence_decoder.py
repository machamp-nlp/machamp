import logging
from typing import Dict

import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics import CategoricalAccuracy
from overrides import overrides

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
        dataset_embeds_dim: int = 0,
        metric: str = "acc",
        **kwargs,
    ) -> None:

        super().__init__(vocab, **kwargs)

        self.task = task
        self.vocab = vocab
        self.input_dim = input_dim + dataset_embeds_dim
        self.loss_weight = loss_weight
        self.metric = metric
        self.num_labels = self.vocab.get_vocab_size(namespace=task)

        self._classifier_input_dim = self.input_dim

        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self.num_labels)
        self._loss = torch.nn.CrossEntropyLoss()

        self._accuracy = CategoricalAccuracy()
        if self.metric == "acc":
            self.metrics = {"acc": CategoricalAccuracy()}
        elif self.metric == "micro-f1":
            self.metrics = {"micro-f1": FBetaMeasure(average='micro')}
        elif self.metric == "macro-f1":
            self.metrics = {"macro-f1": FBetaMeasure(average='macro')}
        else:
            logger.warning(f"ERROR. Metric: {self.metric} unrecognized. Using accuracy instead.")
            self.metrics = {"acc": CategoricalAccuracy()}


    def forward(  # type: ignore
        self, 
        embedded_text: TextFieldTensors, 
        gold_labels: torch.LongTensor = []
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

        output_dict = {"logits": logits, "class_probabilities": probs}

        if gold_labels != None:
            output_dict['loss'] = self._loss(logits, gold_labels.long().view(-1)) * self.loss_weight
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

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        main_metrics = {
            f".run/{self.task}/{metric_name}": metric.get_metric(reset)
            for metric_name, metric in self.metrics.items()
        }
        return {**main_metrics}

