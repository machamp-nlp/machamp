import torch


class UAS:
    def __init__(self):
        self.cor = 0
        self.total = 0
        self.str = 'uas'
        self.metric_scores = {}

    def score(self, pred_heads, pred_rels, gold_heads, gold_rels, mask):
        pred_heads = pred_heads.flatten()
        gold_heads = gold_heads.flatten()

        mask != None:
            mask = torch.flatten(mask)
            pred_heads = pred_heads[mask]
            gold_heads = gold_heads[mask]

        cor_heads = gold_heads.eq(pred_heads)
    
        self.cor += torch.sum(cor_heads).item()
        self.total += len(gold_heads)

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


