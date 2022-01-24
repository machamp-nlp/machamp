import logging
from collections import Counter
from typing import Dict, Union, Optional, List

import sys
import torch
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.training.metrics import CategoricalAccuracy, PearsonCorrelation, SpearmanCorrelation
from overrides import overrides
from machamp.modules import sequence_cross_entropy_with_logits

logger = logging.getLogger(__name__)

@Model.register("machamp_regression_decoder")
class MachampRegressionDecoder(Model):
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
        self.num_labels = 1

        self._classifier_input_dim = self.input_dim

        self._classification_layer = torch.nn.Linear(self._classifier_input_dim, self.num_labels)
        
        self.class_weights = class_weights
        self.loss = torch.nn.MSELoss()

        if self.metric == "pearson":
            self.metrics = {"pearson": PearsonCorrelation()}
        elif self.metric == "spearman":
            self.metrics = {"spearman": SpearmanCorrelation()}
        else:
            print('The regression task-type currently only support pearson as metric')
            sys.exit()


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

        output_dict = {"logits": logits, "class_probabilities": logits}

        if gold_labels != None:
            loss = self.loss(logits.view(logits.shape[0]), gold_labels)
            output_dict['loss'] = loss * self.loss_weight
            for metric in self.metrics.values():
                metric(logits.squeeze(), gold_labels)
        return output_dict


    @overrides
    def make_output_human_readable(
        self, predictions: torch.Tensor
    ) -> List[str]:
        return [str(x) for x in predictions.squeeze().tolist()]

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

