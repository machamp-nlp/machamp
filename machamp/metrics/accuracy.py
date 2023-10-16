import torch


class Accuracy:
    def __init__(self):
        self.cor = 0
        self.total = 0
        self.str = 'accuracy'
        self.metric_scores = {}

    def score(self, preds, golds, vocabulary):
        preds = torch.flatten(preds)
        golds = torch.flatten(golds)

        contents = torch.nonzero(golds != -100)
        # Only unks in gold, probably an indicator that 
        # there are no annotations
        if len(contents) == 0:
            self.total += len(golds)
            return
        preds = preds[contents]
        golds = golds[contents]

        self.total += len(contents)
        self.cor += sum(preds==golds).item()
        

    def reset(self):
        self.cor = 0
        self.total = 0

    def get_score(self):
        if self.total == 0:
            self.metric_scores[self.str] = 0.0
        else:
            self.metric_scores[self.str] = self.cor / self.total
        self.metric_scores["sum"] = self.str
        return self.metric_scores

    def is_active(self):
        return self.total != 0

