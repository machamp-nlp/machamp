import torch
import torch.nn.functional as F

from machamp.model.machamp_decoder import MachampDecoder


class MachampSeqDecoder(MachampDecoder, torch.nn.Module):
    def __init__(
            self,
            task: str,
            vocabulary,
            input_dim: int,
            device: str, 
            decoder_dropout: float = 0.0,
            loss_weight: float = 1.0,
            metric: str = 'accuracy',
            topn: int = 1,
            **kwargs
    ) -> None:
        super().__init__(task, vocabulary, loss_weight, metric, device, **kwargs)

        nlabels = len(self.vocabulary.get_vocab(task))
        self.input_dim = input_dim  # + dec_dataset_embeds_dim
        self.hidden_to_label = torch.nn.Linear(input_dim, nlabels)
        self.hidden_to_label.to(device)
        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=-100)
        self.topn = topn

        self.decoder_dropout = torch.nn.Dropout(decoder_dropout)
        self.decoder_dropout.to(device)

    def forward(self, mlm_out, task_subword_mask, gold=None):
        if self.decoder_dropout.p > 0.0:
            mlm_out =  self.decoder_dropout(mlm_out) 

        logits = self.hidden_to_label(mlm_out)
        out_dict = {'logits': logits}
        if type(gold) != type(None):
            # 0 is the padding/unk label, so skip it for the metric
            maxes = torch.add(torch.argmax(logits[:, :, 1:], 2), 1)
            self.metric.score(maxes, gold, self.vocabulary.inverse_namespaces[self.task])
            if self.additional_metrics:
                for additional_metric in self.additional_metrics:
                    additional_metric.score(maxes, gold, self.vocabulary.inverse_namespaces[self.task])
            flat_length = gold.shape[0] * gold.shape[1]
            loss = self.loss_weight * self.loss_function(logits.view(flat_length, -1), gold.view(flat_length))
            out_dict['loss'] = loss
        return out_dict

    def get_output_labels(self, mlm_out, mask, gold=None):
        """
        logits = batch_size*sent_len*num_labels
        argmax converts to a list of batch_size*sent_len, 
        we add 1 because we leave out the padding/unk 
        token in position 0 (thats what [:,:,1:] does)
        """
        logits = self.forward(mlm_out, mask, gold)['logits']
        if self.topn == 1:
            # 0 is the padding/unk label, so skip it for the metric
            maxes = torch.add(torch.argmax(logits[:, :, 1:], 2), 1)
            return {
                'word_labels': [[self.vocabulary.id2token(token_id, self.task) for token_id in sent] for sent in maxes]}
        else:
            tags = []
            probs = []
            class_probs = F.softmax(logits, -1)
            for sent_scores in class_probs:
                tags.append([])
                probs.append([])
                for word_scores in sent_scores:
                    topk = torch.topk(word_scores[1:], self.topn)
                    tags[-1].append([self.vocabulary.id2token(label_id + 1, self.task) for label_id in topk.indices])
                    probs[-1].append([score.item() for score in topk.values])
            return {'word_labels': tags, 'probs': probs}
