import logging

import torch

from machamp.model.machamp_decoder import MachampDecoder

logger = logging.getLogger(__name__)


class MachampMulticlasDecoder(MachampDecoder, torch.nn.Module):
    def __init__(self, task, vocabulary, input_dim, device, loss_weight: float = 1.0, topn: int = 1,
                 metric: str = 'accuracy', decoder_dropout: float = 0.0, threshold: float = .7, **kwargs):
        super().__init__(task, vocabulary, loss_weight, metric, decoder_dropout, device,
                         **kwargs)

        nlabels = len(self.vocabulary.get_vocab(task))
        self.hidden_to_label = torch.nn.Linear(input_dim, nlabels)
        self.hidden_to_label.to(device)
        self.loss_function = torch.nn.BCEWithLogitsLoss()
        self.topn = topn
        self.threshold = threshold

    def forward(self, mlm_out, mask, gold=None):
        mlm_out = (
            self.decoder_dropout(mlm_out) 
            if self.decoder_dropout.p > 0 else mlm_out
        )
        logits = self.hidden_to_label(mlm_out)
        out_dict = {'logits': logits}
        if type(gold) != type(None):
            gold[gold == -100] = 0
            out_dict['loss'] = self.loss_weight * self.loss_function(logits[:,
                                                                     1:], gold.to(torch.float32)[:,
                                                                          1:])
            
            preds = torch.sigmoid(logits) > self.threshold
            self.metric.score(preds[:,1:], gold.eq(torch.tensor(1.0, device=self.device))[:, 1:], None)
            if self.additional_metrics:
                for additional_metric in self.additional_metrics:
                    additional_metric.score(preds[:,1:], gold.eq(torch.tensor(1.0, device=self.device))[:, 1:], None)
        return out_dict

    def get_output_labels(self, mlm_out, mask, gold=None):
        if self.topn != 1:
            logger.warning("--topn is not implemented for multiclas, as it already can output multiple candidates")
    
        logits = self.forward(mlm_out, mask, gold)['logits']
        all_labels = []
        preds = torch.sigmoid(logits) > self.threshold
        for sent_idx in range(len(preds)):
            sent_labels = []
            for label_idx in range(1, len(preds[sent_idx])):
                if preds[sent_idx][label_idx]:
                    sent_labels.append(self.vocabulary.id2token(label_idx, self.task))
            all_labels.append('|'.join(sent_labels))
        return {'sent_labels': all_labels}

