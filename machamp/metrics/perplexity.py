import math


class Perplexity:
    def __init__(self):
        self.sum = 0
        self.number = 0
        self.str = 'perplexity'
        self.metric_scores = {}

    def score(self, loss):
        self.sum += loss
        self.number += 1

    def reset(self):
        self.sum = 0
        self.number = 0

    def get_score(self):
        if self.sum == 0:
            self.metric_scores[self.str] = 0.0
        self.metric_scores[self.str] = math.exp(self.sum / self.number)
        self.metric_scores["sum"] = self.str
        return self.metric_scores

    def is_active(self):
        return self.number != 0

