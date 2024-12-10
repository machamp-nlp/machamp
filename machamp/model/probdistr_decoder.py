import torch
import torch.nn.functional as F

from machamp.model.machamp_decoder import MachampDecoder


class MachampProbdistributionDecoder(MachampDecoder, torch.nn.Module):
    def __init__(self, task, vocabulary, input_dim, device, loss_weight: float = 1.0, decoder_dropout: float = 0.0, topn: int = 1,
                 metric: str = 'accuracy', **kwargs):
        super().__init__(task, vocabulary, loss_weight, metric, device, **kwargs)

        self.nlabels = len(kwargs['column_idxs'])
        self.hidden_to_label = torch.nn.Linear(input_dim, self.nlabels)
        self.hidden_to_label.to(device)
        self.loss_function = torch.nn.CrossEntropyLoss(reduction='sum', ignore_index=-100)
        # Not sure which loss to use, in the past we used multi_class_cross_entropy_loss and sequence_cross_entropy_with_logits from AllenNLP, 
        # other options: nn.MSELoss nn.BCELoss
        self.topn = topn

        self.decoder_dropout = torch.nn.Dropout(decoder_dropout)
        self.decoder_dropout.to(device)

    def forward(self, mlm_out, mask, gold=None):
        if self.topn != 1:
            logger.warning('topn is not implemented for the probdistr task type, as it is unclear what it should do')

        if self.decoder_dropout.p > 0.0:
            mlm_out =  self.decoder_dropout(mlm_out) 

        logits = self.hidden_to_label(mlm_out)
        out_dict = {'logits': logits}
        if type(gold) != type(None):
            self.metric.score(logits, gold, None, mask)
            if self.additional_metrics:
                for additional_metric in self.additional_metrics:
                    additional_metric.score(maxes, gold, None, self.vocabulary.inverse_namespaces[self.task], None)
            out_dict['loss'] = self.loss_weight * self.loss_function(logits, gold)
        return out_dict

    def get_output_labels(self, mlm_out, mask, gold=None):
        logits = self.forward(mlm_out, mask, gold)['logits']
        # logits is a matrix of batch_size * n_labels
        # however, the predictor expects strings, not sure how to most
        # easily do this,  as they should be written to different columns..
        # probably just using a list and an exception in the predictor 
        # would do the trick

        return {'sent_labels': [str(x.item()) for x in logits]}

