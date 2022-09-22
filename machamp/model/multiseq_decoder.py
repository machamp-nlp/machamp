import torch
import torch.nn.functional as F

from machamp.model.machamp_decoder import MachampDecoder


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
            threshold: float = .0,
            **kwargs
    ) -> None:
        super().__init__(task, vocabulary, loss_weight, metric, device)

        nlabels = len(self.vocabulary.get_vocab(task))
        self.input_dim = input_dim  # + dec_dataset_embeds_dim
        self.hidden_to_label = torch.nn.Linear(input_dim, nlabels)
        self.hidden_to_label.to(device)
        self.loss_function = torch.nn.BCEWithLogitsLoss()
        self.threshold = threshold
        self.topn = topn

    def forward(self, mlm_out, mask, gold=None):
        logits = self.hidden_to_label(mlm_out)
        out_dict = {'logits': logits}
        if type(gold) != type(None):
            # convert scores to binary:
            preds = logits > self.threshold
            self.metric.score(preds[:,:,1:], gold.eq(torch.tensor(1.0, device='cuda:0'))[:,:,1:], mask, self.vocabulary.inverse_namespaces[self.task])
            loss = self.loss_weight * self.loss_function(logits[:,:,1:], gold.to(torch.float32)[:,:,1:])
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
            all_labels = []
            preds = logits > self.threshold
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
        else: # TODO implement topn?
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
