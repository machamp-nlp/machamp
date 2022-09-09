import torch


class Accuracy():
    def __init__(self):
        self.cor = 0
        self.total = 0
        self.str = 'acc.'

    def score(self, preds, golds, mask, vocabulary):
        corrects = preds.eq(golds)
        if len(preds.shape) == 2:
            corrects *= mask
            self.total += torch.sum(mask).item()
        else:
            self.total += len(golds)
        self.cor += torch.sum(corrects).item()

    def reset(self):
        self.cor = 0
        self.total = 0

    def get_score(self):
        if self.total == 0:
            return self.str, 0.0
        return self.str, self.cor / self.total
