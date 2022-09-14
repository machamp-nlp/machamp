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
        super().__init__(task, vocabulary, loss_weight, metric)

        self.input_dim = input_dim  # + dec_dataset_embeds_dim
        self.loss_function = torch.nn.CrossEntropyLoss()

    def forward(self, mlm_preds, gold, mask=None):
        shifted_prediction_scores = mlm_preds[:, :-1, :].contiguous()
        labels = gold[:, 1:].contiguous()

        lm_loss = self.loss_function(shifted_prediction_scores.view(-1, mlm_preds.shape[-1]), labels.view(-1))
        self.metric.score(lm_loss)

        return {'loss': self.loss_weight * lm_loss}

    def get_output_labels(self, mlm_out, mask):
        return {'word_labels': [], 'probs': []}
