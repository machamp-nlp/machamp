import torch
import torch.nn.functional as F

from machamp.model.machamp_decoder import MachampDecoder


class MachampClassificationDecoder(MachampDecoder, torch.nn.Module):
    def __init__(self, task, vocabulary, input_dim, device, loss_weight: float = 1.0, decoder_dropout: float = 0.0, topn: int = 1,
                 metric: str = 'accuracy', **kwargs):
        super().__init__(task, vocabulary, loss_weight, metric, decoder_dropout, device, **kwargs)

        nlabels = len(self.vocabulary.get_vocab(task))
        self.hidden_to_label = torch.nn.Linear(input_dim, nlabels)
        self.hidden_to_label.to(device)
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=-100)
        self.topn = topn

    def forward(self, mlm_out, mask, gold=None):
        if self.decoder_dropout.p > 0.0:
            mlm_out =  self.decoder_dropout(mlm_out) 

        logits = self.hidden_to_label(mlm_out)
        out_dict = {'logits': logits}
        if type(gold) != type(None):
            maxes = torch.add(torch.argmax(logits[:, 1:], 1), 1)
            self.metric.score(maxes, gold, self.vocabulary.inverse_namespaces[self.task])
            if self.additional_metrics:
                for additional_metric in self.additional_metrics:
                    additional_metric.score(maxes, gold, None, self.vocabulary.inverse_namespaces[self.task])
            out_dict['loss'] = self.loss_weight * self.loss_function(logits, gold)
        return out_dict

    def get_output_labels(self, mlm_out, mask, gold=None):
        logits = self.forward(mlm_out, mask, gold)['logits']
        if self.topn == 1:
            maxes = torch.add(torch.argmax(logits[:, 1:], 1), 1)
            return {'sent_labels': [self.vocabulary.id2token(label_id, self.task) for label_id in maxes]}
        else:
            labels = []
            probs = []
            class_probs = F.softmax(logits, -1)
            for sent_scores in class_probs:
                topk = torch.topk(sent_scores[1:], self.topn)
                labels.append([self.vocabulary.id2token(label_id + 1, self.task) for label_id in topk.indices])
                probs.append([score.item() for score in topk.values])
            return {'sent_labels': labels, 'probs': probs}
