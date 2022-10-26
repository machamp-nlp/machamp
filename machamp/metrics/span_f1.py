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

    def score(self, preds, golds, mask, vocabulary_list):
        golds = golds * mask
        preds = preds * mask
        for sent_idx in range(len(golds)):
            gold_labels_str = [vocabulary_list[token] for token in golds[sent_idx] if token != 0]
            pred_labels_str = [vocabulary_list[token] for token in preds[sent_idx] if token != 0]

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

    def get_score(self):
        precision = 0.0 if self.tps + self.fps == 0 else self.tps / (self.tps + self.fps)
        recall = 0.0 if self.tps + self.fns == 0 else self.tps / (self.tps + self.fns)
        f1 = 0.0 if precision + recall == 0 else 2 * (precision * recall) / (precision + recall)
        return self.str, f1
