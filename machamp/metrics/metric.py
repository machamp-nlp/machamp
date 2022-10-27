import logging

from machamp.metrics.accuracy import Accuracy
from machamp.metrics.avg_dist import AvgDist
from machamp.metrics.f1 import F1
from machamp.metrics.las import LAS
from machamp.metrics.multi_accuracy import MultiAccuracy
from machamp.metrics.perplexity import Perplexity
from machamp.metrics.span_f1 import SpanF1

logger = logging.getLogger(__name__)


class Metric:
    def __init__(self, metric_name: str):
        """
        This is a wrapper class that contains a metric (and perhaps 
        in the future multiple metrics). This is mainly included so
        that we won't need a list of if statements in each decoder.

        Parameters
        ----------
        metric_name: str
            The name of the metric, note that exact string matching is used
        """
        self.metrics = {}
        if metric_name == 'accuracy':
            self.metrics[metric_name] = Accuracy()
        elif metric_name == 'multi_acc':
            self.metrics[metric_name] = MultiAccuracy()
        elif metric_name == 'las':
            self.metrics[metric_name] = LAS()
        elif metric_name == 'avg_dist':
            self.metrics[metric_name] = AvgDist()
        elif metric_name == 'perplexity':
            self.metrics[metric_name] = Perplexity()
        elif metric_name == 'f1_binary':
            self.metrics[metric_name] = F1('binary')
        elif metric_name == 'f1_micro':
            self.metrics[metric_name] = F1('micro')
        elif metric_name == 'f1_macro':
            self.metrics[metric_name] = F1('macro')
        elif metric_name == 'span_f1':
            self.metrics[metric_name] = SpanF1()
        else:
            logger.error("metric " + metric_name + ' is not defined in MaChAmp.')
            exit(1)

    def score(self, *kwargs):
        """
        Calculates the variables needed for a specific metric based on prediction
        and gold labels. Note that this accumulates, it is supposed to be called 
        multiple times (once for each batch). The parameters differ per metric 
        (LAS for example needs indices of heads and labels).
        """
        for metric in self.metrics:
            self.metrics[metric].score(*kwargs)

    def reset(self):
        """
        Because the metrics accumulate their internal scores, we need to reset if 
        we want to use the metric again (for example for train and dev split, or
        when having multiple dev sets).
        """
        for metric in self.metrics:
            self.metrics[metric].reset()

    def get_scores(self):
        """
        Return the scores. 
        """
        names = []
        scores = []
        types = []
        for metric in self.metrics:
            name, score = self.metrics[metric].get_score()
            names.append(name)
            scores.append(score)
            types.append(type(self.metrics[metric]))
        return names, scores, types
