import logging

import torch

logger = logging.getLogger(__name__)


class F1:
    def __init__(self, type_f1):
        self.tps = []
        self.fps = []
        self.fns = []
        self.type_f1 = type_f1
        self.str = 'f1_' + type_f1

    def score(self, preds, golds, mask, vocabulary):
        max_label = torch.max(torch.cat((preds, golds)))
        while len(self.tps) <= max_label:
            self.tps.append(0)
            self.fps.append(0)
            self.fns.append(0)

        # A: Check whether to run the evaluation at the token- or sentence-level
        # @TODO: Perhaps there could be better ways to handle this - but this way classification 
        # tasks work (no errors when inspecting golds for "word_idx" indices)
        # R: I changed it  to check the shape of gold, so that it also works when
        # doing token and sent level at the same time
        is_token_level = len(golds.shape) == 2#True if (mask != None) else False
        # TODO, it might be nicer to convert them to a similar shape?

        for sent_idx in range(len(golds)):
            if is_token_level:
                for word_idx in range(len(golds[sent_idx])):
                    if mask[sent_idx][word_idx]:
                        gold = golds[sent_idx][word_idx]
                        pred = preds[sent_idx][word_idx]
                        if gold == pred:
                            self.tps[gold.item()] += 1
                        else:
                            self.fps[pred.item()] += 1
                            self.fns[gold.item()] += 1
            else:
                gold = golds[sent_idx]
                pred = preds[sent_idx]
                if gold == pred:
                    self.tps[gold.item()] += 1
                else:
                    self.fps[pred.item()] += 1
                    self.fns[gold.item()] += 1

    def reset(self):
        self.tps = []
        self.fps = []
        self.fns = []
        self.total = 0

    def get_f1(self, tp, fp, fn):
        precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
        recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
        return 0.0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)

    def get_score(self):
        if self.type_f1 == 'micro':
            return self.str, self.get_f1(sum(self.tps), sum(self.fps), sum(self.fns))

        elif self.type_f1 == 'macro':
            f1s = []
            for label_idx in range(1, len(self.tps)):
                f1s.append(self.get_f1(self.tps[label_idx], self.fps[label_idx], self.fns[label_idx]))
            return self.str, sum(f1s) / len(f1s)

        elif self.type_f1 == 'binary':
            if len(self.tps) > 3:
                logger.error('Choose F1 binary, but there are multiple classes, returning 0.0.')
                return self.str, 0.0
            return self.str, self.get_f1(self.tps[1], self.fps[1], self.fns[1])

        else:
            logger.error('F1 type ' + self.type_f1 + ' not recognized, returning 0.0.')
            return self.str, 0.0
