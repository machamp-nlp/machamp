import torch

from machamp.model.machamp_decoder import MachampDecoder


class MachampRegressionDecoder(MachampDecoder, torch.nn.Module):
    def __init__(self, task, vocabulary, input_dim, device, loss_weight: float = 1.0, topn: int = 1,
                 metric: str = 'avg_dist', decoder_dropout: float = 0.0, **kwargs):
        super().__init__(task, vocabulary, loss_weight, metric, device, **kwargs)

        self.hidden_to_label = torch.nn.Linear(input_dim, 1)
        self.hidden_to_label.to(device)
        self.loss_function = torch.nn.MSELoss()
        self.topn = topn

        self.decoder_dropout = torch.nn.Dropout(decoder_dropout)
        self.decoder_dropout.to(device)

    def forward(self, mlm_out, mask, gold=None):
        if self.topn != 1:
            logger.warning('topn is not implemented for the regression task type, as it is unclear what it should do')
        
        if self.decoder_dropout.p > 0.0:
            mlm_out =  self.decoder_dropout(mlm_out) 

        logits = self.hidden_to_label(mlm_out)
        out_dict = {'logits': logits}
        if type(gold) != type(None):
            self.metric.score(logits, gold, None)
            if self.additional_metrics:
                for additional_metric in self.additional_metrics:
                    additional_metric.score(logits, gold, None)
            loss = self.loss_weight * self.loss_function(logits.flatten(), gold)
            out_dict['loss'] = loss
        return out_dict

    def get_output_labels(self, mlm_out, mask, gold=None):
        logits = self.forward(mlm_out, mask, gold)
        return {'sent_labels': [str(x.item()) for x in logits['logits']]} 

