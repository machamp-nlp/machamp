import torch


class AvgDist:
    def __init__(self):
        self.dists = []
        self.str = 'avg_dist'
        self.metric_scores = {}

    def score(self, preds, golds, vocabulary, mask=None):
        # Following TVD from e.g. https://aclanthology.org/2022.emnlp-main.124.pdf
        dists = torch.abs(preds - golds)/2
        self.dists.extend(torch.sum(dists, dim=1).tolist())

    def reset(self):
        self.dists = []

    def get_score(self):
        if self.dists == []:
            self.metric_scores[self.str] = -1
        else:
            self.metric_scores[self.str] = sum(self.dists) / len(self.dists)
        self.metric_scores["sum"] = self.str
        return self.metric_scores

    def is_active(self):
        return self.dists != []

