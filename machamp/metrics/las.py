import torch


class LAS:
    def __init__(self):
        self.cor = 0
        self.total = 0
        self.str = 'las'
        self.metric_scores = {}

    def score(self, pred_heads, pred_rels, gold_heads, gold_rels, mask):
        pred_rels = pred_rels.flatten()
        gold_rels = gold_rels.flatten()

        pred_heads = pred_heads.flatten()
        gold_heads = gold_heads.flatten()


        if mask != None:
            mask = torch.flatten(mask)
            pred_rels = pred_rels[mask]
            gold_rels = gold_rels[mask]
            pred_heads = pred_heads[mask]
            gold_heads = gold_heads[mask]


        cor_rels = gold_rels.eq(pred_rels)
        cor_heads = gold_heads.eq(pred_heads)
    
        self.cor += torch.sum(cor_rels * cor_heads).item()
        self.total += len(gold_rels)

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


