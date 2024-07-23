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
        self.vocabulary = []
        self.metric_scores = {}

    def score(self, preds, golds, vocabulary, mask):
        self.vocabulary = vocabulary
        # Make sure we have space for counts for all class-labels
        max_label = torch.max(torch.cat((preds, golds)))
        while len(self.tps) <= max_label:
            self.tps.append(0)
            self.fps.append(0)
            self.fns.append(0)

        preds = torch.flatten(preds)
        golds = torch.flatten(golds)
        if mask != None:
            mask = torch.flatten(mask)
            preds = preds[mask]
            golds = golds[mask]

        for gold, pred in zip(golds, preds):
            if gold != -100:
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

    def get_precision(self, tp, fp):
        return 0.0 if tp + fp == 0 else tp / (tp + fp)

    def get_recall(self, tp, fn):
        return 0.0 if tp + fn == 0 else tp / (tp + fn)

    def get_f1(self, precision, recall):
        return 0.0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)

    def get_score(self):
        if self.type_f1 == 'micro':
            precision = self.get_precision(sum(self.tps), sum(self.fps))
            recall = self.get_recall(sum(self.tps), sum(self.fns))
            f1_score = self.get_f1(precision, recall)

            self.metric_scores["precision_" + self.type_f1] = precision
            self.metric_scores["recall_" + self.type_f1] = recall
            self.metric_scores[self.str] = f1_score
            self.metric_scores["sum"] = self.str


        elif self.type_f1 == 'macro':
            f1s = []
            precs = []
            recs = []

            for label_idx in range(1, len(self.tps)):
                label_name = self.vocabulary[label_idx]
                precision = self.get_precision(self.tps[label_idx], self.fps[label_idx])
                recall = self.get_recall(self.tps[label_idx], self.fns[label_idx])
                f1_score = self.get_f1(precision, recall)

                self.metric_scores["precision_" + label_name] = precision
                self.metric_scores["recall_" + label_name] = recall
                self.metric_scores["f1_" + label_name] = f1_score

                f1s.append(f1_score)
                precs.append(precision)
                recs.append(recall)

            if precs == []:
                self.metric_scores[self.str] = 0.0
            else:
                self.metric_scores["precision_" + self.type_f1] = sum(precs) / len(precs)
                self.metric_scores["recall_" + self.type_f1] = sum(recs) / len(recs)
                self.metric_scores[self.str] = sum(f1s) / len(f1s)
            self.metric_scores["sum"] = self.str

        elif self.type_f1 == 'binary':
            if len(self.tps) > 3:
                logger.error('Choose F1 binary, but there are multiple classes, returning 0.0.')

                self.metric_scores["precision_" + self.type_f1] = 0.0
                self.metric_scores["recall_" + self.type_f1] = 0.0
                self.metric_scores[self.str] = 0.0
                self.metric_scores["sum"] = self.str

            else:   
                if len(self.tps) > 0 or len(self.fps) > 0 or len(self.fns) > 0:
                    precision = self.get_precision(self.tps[1], self.fps[1])
                    recall = self.get_recall(self.tps[1], self.fns[1])
                    f1_score = self.get_f1(precision, recall)
                else:
                    precision = 0.0
                    recall = 0.0
                    f1 = 0.0

                self.metric_scores["precision_" + self.type_f1] = precision
                self.metric_scores["recall_" + self.type_f1] = recall
                self.metric_scores[self.str] = f1_score
                self.metric_scores["sum"] = self.str

        else:
            logger.error('F1 type ' + self.type_f1 + ' not recognized, returning 0.0.')

            self.metric_scores["precision_" + self.type_f1] = 0.0
            self.metric_scores["recall_" + self.type_f1] = 0.0
            self.metric_scores[self.str] = 0.0
            self.metric_scores["sum"] = self.str
        return self.metric_scores

    def is_active(self):
        return len(self.tps) + len(self.fps) + len(self.fns) != 0
