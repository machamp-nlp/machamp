from typing import Dict, Optional, List, Any

from overrides import overrides
import torch
import numpy
import logging

from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward, Seq2SeqEncoder
from allennlp.nn import InitializerApplicator
from allennlp.training.metrics import CategoricalAccuracy
from machamp.metrics.fbeta_measure import FBetaMeasure

logger = logging.getLogger(__name__)


@Model.register("machamp_sentence_classifier")
class BasicClassifier(Model):

    def __init__(
        self,
        vocab: Vocabulary,
        task: str,
        encoder: Seq2SeqEncoder,
        feedforward: Optional[FeedForward] = None,
        prev_task: str = None,
        prev_task_embed_dim: int = None,
        label_smoothing: float = 0.0,
        adaptive: bool = False,
        loss_weight: float = 1.0,
        dropout: float = None,
        label_namespace: str = "labels",
        metric: str = 'acc',
        task_types: List[str] = None,
        tasks: List[str] = None,
        initializer: InitializerApplicator = InitializerApplicator(),
        **kwargs,
    ) -> None:

        super().__init__(vocab)

        self.task = task
        self.encoder = encoder
        self.metric = metric

        self._feedforward = feedforward
        if feedforward is not None:
            self._classifier_input_dim = self._feedforward.get_output_dim()
        else:
            self._classifier_input_dim = self.encoder.get_output_dim()

        if dropout:
            self._dropout = torch.nn.Dropout(dropout)
        else:
            self._dropout = None
        self._label_namespace = label_namespace

        self.num_classes = self.vocab.get_vocab_size(task)

        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self.num_classes)
        self.loss_weight = loss_weight

        if self.metric == "acc":
            self.metrics = {"acc": CategoricalAccuracy()}
        elif self.metric == "micro-f1":
            self.metrics = {"micro-f1": FBetaMeasure(average='micro')}
        elif self.metric == "macro-f1":
            self.metrics = {"macro-f1": FBetaMeasure(average='macro')}
        else:
            logger.warning(f"ERROR. Metric: {self.metric} unrecognized. Using accuracy instead.")
            self.metrics = {"acc": CategoricalAccuracy()}



        self._loss = torch.nn.CrossEntropyLoss()
        initializer(self)

    @overrides
    def forward(self,
                encoded_text: torch.FloatTensor,
                mask: torch.LongTensor,
                gold_tags: Dict[str, torch.LongTensor],
                prev_task_classes: torch.LongTensor = None,
                metadata: List[Dict[str, Any]] = None) -> Dict[str, torch.Tensor]:


        hidden = encoded_text
        embedded_text = self.encoder(hidden)

        if self._dropout:
            embedded_text = self._dropout(embedded_text)

        if self._feedforward is not None:
            embedded_text = self._feedforward(embedded_text)

        batch_size, _ = embedded_text.size()
        output_dim = [batch_size, self.num_classes]

        logits = self._classification_layer(embedded_text)
        probs = torch.nn.functional.softmax(logits, dim=-1)

        output_dict = {"logits": logits, "class_probabilities": probs}

        # gold_label per sentence
        if self.task in gold_tags:
            gold_tags = gold_tags.get(self.task, None)[:, 0]
            loss = self._loss(logits, gold_tags.view(-1))
            output_dict["loss"] = self.loss_weight * loss

            for metric in self.metrics.values():
                metric(logits, gold_tags)

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        all_words = output_dict["words"]
        all_predictions = output_dict["class_probabilities"][self.task].cpu().data.numpy()
        if all_predictions.ndim == 3:
            predictions_list = [all_predictions[i] for i in range(all_predictions.shape[0])]
        else:
            predictions_list = [all_predictions]
        all_tags = []
        for predictions in predictions_list:
            argmax_indices = numpy.argmax(predictions, axis=-1)
            tag = self.vocab.get_token_from_index(argmax_indices[0], namespace=self.task)
        all_tags.append(tag)

        # if it needs to be token level:
        output_dict[self.task] = [all_tags*len(all_words[0])]

        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:

        main_metrics = {
            f".run/{self.task}/{metric_name}": metric.get_metric(reset)
            for metric_name, metric in self.metrics.items()
        }

        return {**main_metrics}
