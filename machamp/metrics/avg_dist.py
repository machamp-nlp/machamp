import torch


class AvgDist:
    def __init__(self):
        self.dists = []
        self.str = 'avg_dist'
        self.metric_scores = {}

    def score(self, preds, golds, vocabulary):
        self.dists.extend(torch.abs(preds.flatten() - golds.flatten()).tolist())

    def reset(self):
        self.dists = []

    def get_score(self):
        if self.dists == []:
            self.metric_scores[self.str] = -1
        else:
            self.metric_scores[self.str] = sum(self.dists) / len(self.dists)
        self.metric_scores["sum"] = self.str
        return self.metric_scores
