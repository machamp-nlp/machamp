import torch
import torch.nn.functional as F

from machamp.model.machamp_decoder import MachampDecoder


class MachampMulticlasDecoder(MachampDecoder, torch.nn.Module):
    def __init__(self, task, vocabulary, input_dim, device, loss_weight: float = 1.0, topn: int = 1,
                 metric: str = 'accuracy', threshold: float = .0, **kwargs):
        super().__init__(task, vocabulary, loss_weight, metric, device)

        nlabels = len(self.vocabulary.get_vocab(task))
        self.hidden_to_label = torch.nn.Linear(input_dim, nlabels)
        self.hidden_to_label.to(device)
        self.loss_function = torch.nn.BCEWithLogitsLoss()
        self.topn = topn
        self.threshold = threshold

    def forward(self, mlm_out, mask, gold=None):
        logits = self.hidden_to_label(mlm_out)
        out_dict = {'logits': logits}
        if type(gold) != type(None):
            preds = logits > self.threshold
            self.metric.score(preds[:,1:], gold.eq(torch.tensor(1.0, device=self.device))[:,1:], mask, None)
            out_dict['loss'] = self.loss_weight * self.loss_function(logits[:,1:], gold.to(torch.float32)[:,1:])
        return out_dict

    def get_output_labels(self, mlm_out, mask, gold=None):
        logits = self.forward(mlm_out, mask, gold)['logits']
        if self.topn == 1:
            all_labels = []
            preds = logits > self.threshold
            for sent_idx in range(len(preds)):
                sent_labels = []
                for label_idx in range(1,len(preds[sent_idx])):
                    if preds[sent_idx][label_idx]:
                        sent_labels.append(self.vocabulary.id2token(label_idx, self.task))
                all_labels.append('|'.join(sent_labels))
            return {'sent_labels': all_labels}

        else: # TODO implement top-n
            labels = []
            probs = []
            class_probs = F.softmax(logits, -1)
            for sent_scores in class_probs:
                topk = torch.topk(sent_scores[1:], self.topn)
                labels.append([self.vocabulary.id2token(label_id + 1, self.task) for label_id in topk.indices])
                probs.append([score.item() for score in topk.values])
            return {'sent_labels': labels, 'probs': probs}
