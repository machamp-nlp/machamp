import torch

from machamp.model.machamp_decoder import MachampDecoder


class MachampRegressionDecoder(MachampDecoder, torch.nn.Module):
    def __init__(self, task, vocabulary, input_dim, device, loss_weight: float = 1.0, topn: int = 1,
                 metric: str = 'avg_dist', **kwargs):
        super().__init__(task, vocabulary, loss_weight, metric, device, **kwargs)

        self.hidden_to_label = torch.nn.Linear(input_dim, 1)
        self.hidden_to_label.to(device)
        self.loss_function = torch.nn.MSELoss()
        self.topn = topn

    def forward(self, mlm_out, mask, gold=None):
        if self.topn != 1:
            logger.warning('topn is not implemented for the regression task type, as it is unclear what it should do')
        logits = self.hidden_to_label(mlm_out)
        out_dict = {'logits': logits}
        if type(gold) != type(None):
            self.metric.score(logits, gold, mask, None)
            loss = self.loss_weight * self.loss_function(logits.flatten(), gold)
            out_dict['loss'] = loss
        return out_dict

    def get_output_labels(self, mlm_out, mask, gold=None):
        logits = self.forward(mlm_out, mask, gold)
        return {'sent_labels': [str(x.item()) for x in logits]} 

