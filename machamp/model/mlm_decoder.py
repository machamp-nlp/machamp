import torch

from machamp.model.machamp_decoder import MachampDecoder


class MachampLMDecoder(MachampDecoder, torch.nn.Module):
    def __init__(
            self,
            task: str,
            vocabulary,
            input_dim: int,
            device: str,
            loss_weight: float = 1.0,
            metric: str = 'accuracy',
            topn: int = 1,
            **kwargs
    ) -> None:
        super().__init__(task, vocabulary, loss_weight, metric, device, **kwargs)

        self.input_dim = input_dim  # + dec_dataset_embeds_dim
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=0)
        self.topn = topn

    def forward(self, mlm_preds, gold, mask=None):        
        size = gold.shape[0] * gold.shape[1]
        pred_input = mlm_preds.reshape(size, mlm_preds.shape[-1])
        lm_loss = self.loss_function(pred_input, gold.view(size))
        self.metric.score(lm_loss)

        return {'loss': self.loss_weight * lm_loss}

    def get_output_labels(self, mlm_out, gold, mask=None):
        # Not sure what to return here?
        self.forward(mlm_out, gold, mask)
        return {'word_labels': [], 'probs': []}
