import logging

logger = logging.getLogger(__name__)


def to_spans(tags):
    spans = set()
    for beg in range(len(tags)):
        if tags[beg][0] not in 'BIO':
            logger.error("Warning, one of your labels is not following the BIO scheme: " + tags[
                beg] + " the span-f1 will not be calculated correctly")
        if tags[beg][0] == 'B':
            end = beg
            for end in range(beg + 1, len(tags)):
                if tags[end][0] != 'I':
                    break
            spans.add(str(beg) + '-' + str(end) + ':' + tags[beg][2:])
    return spans


class SpanF1:
    def __init__(self):
        self.tps = 0
        self.fps = 0
        self.fns = 0
        self.str = 'span_f1'
        self.metric_scores = {}

    def score(self, preds, golds, vocabulary_list):
        golds[golds == -100] = 0
        for sent_idx in range(len(golds)):
            if 0 in golds[sent_idx]:
                length = (golds[sent_idx]==0).nonzero(as_tuple=False)[0].item()
            else:
                length = len(golds[sent_idx])
            gold_labels_str = [vocabulary_list[token] for token in golds[sent_idx][:length]]
            pred_labels_str = [vocabulary_list[token] for token in preds[sent_idx][:length]]

            spans_gold = to_spans(gold_labels_str)
            spans_pred = to_spans(pred_labels_str)
            overlap = len(spans_gold.intersection(spans_pred))
            self.tps += overlap
            self.fps += len(spans_pred) - overlap
            self.fns += len(spans_gold) - overlap

    def reset(self):
        self.tps = 0
        self.fps = 0
        self.fns = 0

    def get_precision(self, tp, fp):
        return 0.0 if tp + fp == 0 else tp / (tp + fp)

    def get_recall(self, tp, fn):
        return 0.0 if tp + fn == 0 else tp / (tp + fn)

    def get_f1(self, precision, recall):
        return 0.0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)

    def get_score(self):
        precision = self.get_precision(self.tps, self.fps)
        recall = self.get_recall(self.tps, self.fns)
        f1_score = self.get_f1(precision, recall)

        self.metric_scores["precision"] = precision
        self.metric_scores["recall"] = recall
        self.metric_scores[self.str] = f1_score
        self.metric_scores["sum"] = self.str
        return self.metric_scores

    def is_active(self):
        return self.tps + self.fps + self.fns  != 0

