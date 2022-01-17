import torch
from allennlp.training.metrics.average import Average
from allennlp.training.metrics.metric import Metric
from overrides import overrides
import math

@Metric.register("multi_accuracy")
class MultiAccuracy(Metric):
    def __init__(self) -> None:
        self._total = 0
        self._correct = 0

    @overrides
    def __call__(self, predictions, gold_labels):
        """
        # Parameters

        value : `float`
            The value to average.
        """
        words_pred = predictions.view(predictions.shape[0]*predictions.shape[1], -1)
        words_gold = gold_labels.view(gold_labels.shape[0]*gold_labels.shape[1], -1)
        results = torch.all(words_gold.eq(words_pred), dim=1)
        self._correct += sum(results.tolist())
        self._total += len(results)


    @overrides
    def get_metric(self, reset: bool = False):
        """
        # Returns

        The average of all values that were passed to `__call__`.
        """

        return 0.0 if self._total == 0 else self._correct/self._total 

    @overrides
    def reset(self):
        self._total = 0
        self._correct = 0

