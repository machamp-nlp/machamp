import torch


class AvgDist:
    def __init__(self):
        self.dists = []
        self.str = 'avg_dist.'

    def score(self, preds, golds, mask, vocabulary):
        self.dists.extend(torch.abs(preds.flatten() - golds.flatten()).tolist())

    def reset(self):
        self.dists = []

    def get_score(self):
        if self.dists == []:
            return self.str, -1
        return self.str, sum(self.dists) / len(self.dists)
