import logging
from typing import cast, List

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

from machamp.model.machamp_decoder import MachampDecoder
from machamp.modules.allennlp.conditional_random_field import ConditionalRandomField, allowed_transitions


class MachampCRFDecoder(MachampDecoder, torch.nn.Module):
    def __init__(
            self,
            task: str,
            vocabulary,
            input_dim: int,
            device: str,
            loss_weight: float = 1.0,
            metric: str = 'accuracy',
            decoder_dropout: float = 0.0,
            topn: int = 1,
            **kwargs
    ) -> None:
        super().__init__(task, vocabulary, loss_weight, metric, device, **kwargs)

        nlabels = len(self.vocabulary.get_vocab(task))
        self.input_dim = input_dim  # + dec_dataset_embeds_dim
        self.hidden_to_label = torch.nn.Linear(input_dim, nlabels)
        self.hidden_to_label.to(device)

        self.decoder_dropout = torch.nn.Dropout(decoder_dropout)
        self.decoder_dropout.to(device)

        # hardcoded for now
        constraints = allowed_transitions('BIO', vocabulary.inverse_namespaces[task])
        self.crf_layer = ConditionalRandomField(
            nlabels, constraints
        )

        self.loss_function = torch.nn.CrossEntropyLoss(ignore_index=0)
        if topn != 1:
            logger.info(
                "Top-n for crf is not supported for now, as it is unclear how to get the probabilities. We disabled "
                "it automatically")
            topn = 1
        self.topn = topn

    def forward(self, mlm_out, mask, gold=None):
        if self.decoder_dropout.p > 0.0:
            mlm_out =  self.decoder_dropout(mlm_out) 

        logits = self.hidden_to_label(mlm_out)
        best_paths = self.crf_layer.viterbi_tags(logits, mask)

        predicted_tags = cast(List[List[int]], [x[0] for x in best_paths])
        out_dict = {'logits': logits}
        # map mask to special token, to avoid out of bounds
        gold[gold==-100] = 0

        if type(gold) != type(None):
            log_likelihood = self.crf_layer.forward(logits, gold, mask)

            # Represent viterbi tags as "class probabilities" that we can
            # feed into the metrics
            class_probabilities = logits * 0.0
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1
            maxes = torch.add(torch.argmax(class_probabilities[:, :, 1:], 2), 1)
            self.metric.score(maxes, gold, self.vocabulary.inverse_namespaces[self.task])
            if self.additional_metrics:
                for additional_metric in self.additional_metrics:
                    additional_metric.score(maxes, gold, self.vocabulary.inverse_namespaces[self.task])
            out_dict['loss'] = -log_likelihood * self.loss_weight
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
            best_paths = self.crf_layer.viterbi_tags(logits, mask)

            predicted_tags = cast(List[List[int]], [x[0] for x in best_paths])

            class_probabilities = logits * 0.0
            for i, instance_tags in enumerate(predicted_tags):
                for j, tag_id in enumerate(instance_tags):
                    class_probabilities[i, j, tag_id] = 1
            maxes = torch.add(torch.argmax(class_probabilities[:, :, 1:], 2), 1)
            return {
                'word_labels': [[self.vocabulary.id2token(token_id, self.task) for token_id in sent] for sent in maxes]}
        else:
            best_paths = self.crf_layer.viterbi_tags(logits, mask, top_k=self.topn)
            # for 2 sentences with topn==1, it looks like:
            # [([3, 4, 3, 4, 4, 3, 4, 3, 4, 2, 1, 4, 2, 1, 2, 4, 2, 1, 4], 11.956195831298828), ([4, 3, 4, 3, 4, 3, 4,
            # 3, 4, 1, 4, 3, 4, 4, 5, 5, 5, 5, 1, 1, 4, 3, 4], 13.938811302185059)]

            labels = []
            probs = []
            for sent in best_paths:
                labels.append([])
                probs.append([])
                normed_probs = F.softmax(torch.tensor([x[1] for x in sent]), -1).tolist()
                for word_idx in range(len(sent[0][0])):
                    labels[-1].append([])
                    probs[-1].append([])
                    for n in range(self.topn):
                        labels[-1][-1].append(self.vocabulary.id2token(sent[n][0][word_idx], self.task))
                        probs[-1][-1].append(normed_probs[n])
            return {'word_labels': labels, 'probs': probs}
