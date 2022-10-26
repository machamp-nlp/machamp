import torch


class LAS:
    def __init__(self):
        self.cor = 0
        self.total = 0
        self.str = 'las'

    def score(self, pred_heads, pred_rels, gold_heads, gold_rels, mask):
        mask = mask.flatten()
        pred_rels = pred_rels.flatten()
        gold_rels = gold_rels.flatten()
        cor_rels = gold_rels.eq(pred_rels)

        pred_heads = pred_heads.flatten()
        gold_heads = gold_heads.flatten()
        cor_heads = gold_heads.eq(pred_heads)

        self.cor += torch.sum(cor_rels * cor_heads * mask).item()
        self.total += torch.sum(mask).item()

    def reset(self):
        self.cor = 0
        self.total = 0

    def get_score(self):
        if self.total == 0:
            return self.str, 0.0
        return self.str, self.cor / self.total
