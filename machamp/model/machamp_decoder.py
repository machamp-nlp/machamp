import torch

from machamp.metrics.metric import Metric


class MachampDecoder(torch.nn.Module):
    def __init__(self, task, vocabulary, loss_weight: float = 1.0, metric: str = 'avg_dist'):
        super().__init__()

        self.task = task
        self.vocabulary = vocabulary
        self.metric = Metric(metric)
        self.loss_weight = loss_weight

    def reset_metrics(self):
        self.metric.reset()

    def get_metrics(self):
        return self.metric.get_scores()
