import logging
import torch

logger = logging.getLogger(__name__)

from machamp.metrics.metric import Metric


class MachampDecoder(torch.nn.Module):
    def __init__(self, task, vocabulary, loss_weight: float = 1.0, metric: str = 'avg_dist',  device: str = 'cpu', **kwargs):
        super().__init__()

        self.task = task
        self.vocabulary = vocabulary
        self.metric = Metric(metric)
        self.loss_weight = loss_weight
        self.device = device
        
        if "additional_metrics" in kwargs:
            if type(kwargs["additional_metrics"]) == str:
                self.additional_metrics = [Metric(kwargs["additional_metrics"])]
            elif type(kwargs["additional_metrics"]) == list:
                self.additional_metrics = [Metric(m) for m in kwargs["additional_metrics"]]
            else:
                logger.error('Error, additional_metrics ' + str(kwargs["additional_metrics"]) + ' is not a string nor a list of strings')
                exit(1)
        else:
            self.additional_metrics = None

    def reset_metrics(self):
        self.metric.reset()
        if self.additional_metrics:
            for additional_metric in self.additional_metrics:
                additional_metric.reset()

    def get_metrics(self):
        metric_scores = self.metric.get_scores()
        if self.additional_metrics:
            for additional_metric in self.additional_metrics:
                additional_metric_scores = additional_metric.get_scores()
                for key, value in additional_metric_scores.items():
                    del value["sum"] # to ensure only the main metric is used
                    metric_scores[key] = value
        return metric_scores
