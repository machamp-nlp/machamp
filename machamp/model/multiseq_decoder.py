import logging

import torch

from machamp.model.machamp_decoder import MachampDecoder

logger = logging.getLogger(__name__)

class MachampMultiseqDecoder(MachampDecoder, torch.nn.Module):
    def __init__(
            self,
            task: str,
            vocabulary,
            input_dim: int,
            device: str,
            loss_weight: float = 1.0,
            metric: str = 'accuracy',
            topn: int = 1,
            threshold: float = .7,
            **kwargs
    ) -> None:
        super().__init__(task, vocabulary, loss_weight, metric, device, **kwargs)

        nlabels = len(self.vocabulary.get_vocab(task))
        self.input_dim = input_dim  # + dec_dataset_embeds_dim
        self.hidden_to_label = torch.nn.Linear(input_dim, nlabels)
        self.hidden_to_label.to(device)
        # We do not reduce the loss, as we have to mask it (or ignore_idx), which does not exists for BCE:
        # https://discuss.pytorch.org/t/question-about-bce-losses-interface-and-features/50969/6
        self.loss_function = torch.nn.BCEWithLogitsLoss(reduction='none')
        self.threshold = threshold
        self.topn = topn

    def forward(self, mlm_out, mask, gold=None):
        logits = self.hidden_to_label(mlm_out)
        out_dict = {'logits': logits}
        if type(gold) != type(None):
            gold[gold==-100] = 0
            # There is no ignore_index nor a mask option, so we use the mask as weights (multiply by 1 or 0)
            # Hence, we have to reshape the mask to match the labels as well.
            loss = self.loss_function.forward(logits[:, :, 1:], gold.to(torch.float32)[:, :, 1:])
            mask = mask[:, :, None]
            mask.expand(-1,-1, loss.shape[-1])
            loss = loss * mask
            loss = self.loss_weight * torch.mean(loss)

            preds = torch.sigmoid(logits) > self.threshold
            self.metric.score(preds[:, :, 1:], gold.eq(torch.tensor(1.0, device='cuda:0'))[:, :, 1:], mask,
                              self.vocabulary.inverse_namespaces[self.task])
            if self.additional_metrics:
                for additional_metric in self.additional_metrics:
                    additional_metric.score(preds[:, :, 1:], gold.eq(torch.tensor(1.0, device='cuda:0'))[:, :, 1:], mask,
                              self.vocabulary.inverse_namespaces[self.task])
            out_dict['loss'] = loss
        return out_dict

    def get_output_labels(self, mlm_out, mask, gold=None):
        """
        logits = batch_size*sent_len*num_labels
        argmax converts to a list of batch_size*sent_len, 
        we add 1 because we leave out the padding/unk 
        token in position 0 (thats what [:,:,1:] does)
        """
        if self.topn != 1:
            logger.warning("--topn is not implemented for multiseq, as it already can output multiple candidates")

        logits = self.forward(mlm_out, mask, gold)['logits']
        preds = torch.sigmoid(logits) > self.threshold
        all_labels = []
        for sent_idx in range(len(preds)):
            sent_labels = []
            for word_idx in range(len(preds[sent_idx])):
                word_labels = []
                for label_idx in range(1, len(preds[sent_idx][word_idx])):
                    if preds[sent_idx][word_idx][label_idx]:
                        word_labels.append(self.vocabulary.id2token(label_idx, self.task))
                sent_labels.append('|'.join(word_labels))
            all_labels.append(sent_labels)
        return {'word_labels': all_labels}

