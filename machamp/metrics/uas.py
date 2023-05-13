import torch


class LAS:
    def __init__(self):
        self.cor = 0
        self.total = 0
        self.str = 'las'
        self.metric_scores = {}

    def score(self, pred_heads, pred_rels, gold_heads, gold_rels):
        pred_heads = pred_heads.flatten()
        gold_heads = gold_heads.flatten()
        cor_heads = gold_heads.eq(pred_heads)
    
        mask = gold_rels != -100
        self.cor += torch.sum(cor_heads * mask).item()
        self.total += torch.sum(mask).item()

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


