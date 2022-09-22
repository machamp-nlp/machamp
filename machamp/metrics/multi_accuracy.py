import torch


class MultiAccuracy():
    def __init__(self):
        self.cor = 0
        self.total = 0
        self.str = 'multi-acc.'

    def score(self, preds, golds, mask, vocabulary):
        # TODO: can this be done more efficient?
        if len(preds.shape) == 3:
            for sent_idx in range(len(mask)):
                for word_idx in range(len(mask[sent_idx])):
                    if mask[sent_idx][word_idx]:
                        if torch.all(preds[sent_idx][word_idx] == golds[sent_idx][word_idx]):
                            self.cor += 1
                        self.total += 1
        if len(preds.shape) == 2:
            for sent_idx in range(len(preds)):
                if torch.all(preds[sent_idx] == golds[sent_idx]):
                    self.cor += 1
                self.total += 1

    def reset(self):
        self.cor = 0
        self.total = 0

    def get_score(self):
        if self.total == 0:
            return self.str, 0.0
        return self.str, self.cor / self.total
