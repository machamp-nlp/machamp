import torch
from allennlp.training.metrics.average import Average
from allennlp.training.metrics.metric import Metric
from overrides import overrides
import math

@Metric.register("perplexity_fixed")
class Perplexity(Average):
    """
    Perplexity is a common metric used for evaluating how well a language model
    predicts a sample.

    Notes
    -----
    Assumes negative log likelihood loss of each batch (base e). Provides the
    average perplexity of the batches.
    """

    @overrides
    def get_metric(self, reset: bool = False):
        """
        # Returns

        The accumulated perplexity.
        """
        average_loss = super().get_metric(reset)
        if average_loss == 0:
            perplexity = 0.0
        # R: FIXED HERE (added else), should put in a pull request
        else:
            # Exponentiate the loss to compute perplexity
            if type(average_loss) == float:
                perplexity = math.exp(average_loss)
            else:
                perplexity = float(torch.exp(average_loss))

        return perplexity
