import torch
# This could better be converted to be fuly in torch, or fully in native python

def get_std_scores(x):
    standard_score_x = []
    mean_x = sum(x)/len(x)
    standard_deviation_x = torch.std(torch.tensor(x))
    for observation in x:
        standard_score_x.append((observation - mean_x)/standard_deviation_x)

    return standard_score_x

class Pearson:
    def __init__(self):
        self.x = []
        self.y = []
        self.str = 'pearson'
        self.metric_scores = {}

    def score(self, preds, golds, vocabulary, mask=None):
        self.x.extend(preds.flatten().tolist())
        self.y.extend(golds.flatten().tolist())

    def reset(self):
        self.dists = []

    def get_score(self):
        if self.x == []:
            self.metric_scores[self.str] = -1
        else:
            #torch.corrcoef(torch.tensor([x,y])) # a bit new
            n = len(self.y)
            standard_score_x = get_std_scores(self.x)
            standard_score_y = get_std_scores(self.y)
  
            pearson = (sum([i*j for i,j in zip(standard_score_x, standard_score_y)]))/(n-1)
            self.metric_scores[self.str] = pearson.item()
        self.metric_scores["sum"] = self.str
        return self.metric_scores

    def is_active(self):
        return self.x != []

