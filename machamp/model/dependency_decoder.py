## This dependency head is mainly based on the Allennlp implementation: 
## https://github.com/allenai/allennlp-models/blob/main/allennlp_models/structured_prediction/models/biaffine_dependency_parser.py

import copy
import logging
from typing import Dict, Tuple

import numpy
import torch
import torch.nn.functional as F

from machamp.model.machamp_decoder import MachampDecoder
from machamp.modules.allennlp.bilinear_matrix_attention import BilinearMatrixAttention
from machamp.modules.allennlp.chu_liu_edmonds import decode_mst

logger = logging.getLogger(__name__)


def get_range_vector(size: int, device: int) -> torch.Tensor:
    """
    Returns a range vector with the desired size, starting at 0. The CUDA implementation
    is meant to avoid copy data from CPU to GPU.
    """
    if device > -1:
        return torch.cuda.LongTensor(size, device=device).fill_(1).cumsum(0) - 1
    else:
        return torch.arange(0, size, dtype=torch.long)


def masked_log_softmax(vector: torch.Tensor, mask: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    `torch.nn.functional.log_softmax(vector)` does not work if some elements of `vector` should be
    masked.  This performs a log_softmax on just the non-masked portions of `vector`.  Passing
    `None` in for the mask is also acceptable; you'll just get a regular log_softmax.

    `vector` can have an arbitrary number of dimensions; the only requirement is that `mask` is
    broadcastable to `vector's` shape.  If `mask` has fewer dimensions than `vector`, we will
    unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
    do it yourself before passing the mask into this function.

    In the case that the input vector is completely masked, the return value of this function is
    arbitrary, but not `nan`.  You should be masking the result of whatever computation comes out
    of this in that case, anyway, so the specific values returned shouldn't matter.  Also, the way
    that we deal with this case relies on having single-precision floats; mixing half-precision
    floats with fully-masked vectors will likely give you `nans`.

    If your logits are all extremely negative (i.e., the max value in your logit vector is -50 or
    lower), the way we handle masking here could mess you up.  But if you've got logit values that
    extreme, you've got bigger problems than this.
    """
    if mask is not None:
        while mask.dim() < vector.dim():
            mask = mask.unsqueeze(1)
        # vector + mask.log() is an easy way to zero out masked elements in logspace, but it
        # results in nans when the whole vector is masked.  We need a very small value instead of a
        # zero in the mask for these cases.
        vector = vector + (mask + 1e-13).log()
    return torch.nn.functional.log_softmax(vector, dim=dim)


class MachampDepDecoder(MachampDecoder, torch.nn.Module):
    """
    This dependency parser follows the model of
    [Deep Biaffine Attention for Neural Dependency Parsing (Dozat and Manning, 2016)]
    (https://arxiv.org/abs/1611.01734) .

    Word representations are generated using a bidirectional LSTM,
    followed by separate biaffine classifiers for pairs of words,
    predicting whether a directed arc exists between the two words
    and the dependency label the arc should have. Decoding can either
    be done greedily, or the optimal Minimum Spanning Tree can be
    decoded using Edmond's algorithm by viewing the dependency tree as
    a MST on a fully connected graph, where nodes are words and edges
    are scored dependency arcs.

    # Parameters

    vocab : `MachampVocabulary`, required
        A MachampVocabulary, required in order to compute sizes for input/output projections.
    text_field_embedder : `TextFieldEmbedder`, required
        Used to embed the `tokens` `TextField` we get as input to the model.
    encoder : `Seq2SeqEncoder`
        The encoder (with its own internal stacking) that we will use to generate representations
        of tokens.
    tag_representation_dim : `int`, required.
        The dimension of the MLPs used for dependency tag prediction.
    arc_representation_dim : `int`, required.
        The dimension of the MLPs used for head arc prediction.
    tag_feedforward : `FeedForward`, optional, (default = `None`).
        The feedforward network used to produce tag representations.
        By default, a 1 layer feedforward network with an elu activation is used.
    arc_feedforward : `FeedForward`, optional, (default = `None`).
        The feedforward network used to produce arc representations.
        By default, a 1 layer feedforward network with an elu activation is used.
        If false, decoding is greedy.
    dropout : `float`, optional, (default = `0.0`)
        The variational dropout applied to the output of the encoder and MLP layers.
    """

    def __init__(
            self,
            task: str,
            vocabulary,
            input_dim: int,
            device: str,
            loss_weight: float = 1.0,
            metric: str = 'las',
            decoder_dropout: float = 0.0,
            topn: int = 1,
            tag_representation_dim: int = 256,
            arc_representation_dim: int = 768,
            **kwargs
    ) -> None:
        super().__init__(task, vocabulary, loss_weight, metric, decoder_dropout, device, **kwargs)

        self.input_dim = input_dim  # + dec_dataset_embeds_dim
        arc_representation_dim = arc_representation_dim  # + dec_dataset_embeds_dim

        self.head_arc_feedforward = torch.nn.Linear(self.input_dim, arc_representation_dim).to(self.device)
        self.child_arc_feedforward = copy.deepcopy(self.head_arc_feedforward)

        self.arc_attention = BilinearMatrixAttention(
            arc_representation_dim, arc_representation_dim, use_input_biases=True
        ).to(self.device)

        num_labels = len(self.vocabulary.get_vocab(task))
        self.topn = topn

        self.head_tag_feedforward = torch.nn.Linear(self.input_dim, tag_representation_dim).to(self.device)

        self.child_tag_feedforward = copy.deepcopy(self.head_tag_feedforward)

        self.tag_bilinear = torch.nn.modules.Bilinear(
            tag_representation_dim, tag_representation_dim, num_labels
        ).to(self.device)

        self._head_sentinel = torch.nn.Parameter(torch.randn([1, 1, self.input_dim], device=self.device))

    def forward(
            self,  # type: ignore
            embedded_text: torch.LongTensor,
            mask: torch.LongTensor,
            golds = None
    ) -> Dict[str, torch.Tensor]:
        """
        # Parameters

        words : `TextFieldTensors`, required
            The output of `TextField.as_array()`, which should typically be passed directly to a
            `TextFieldEmbedder`. This output is a dictionary mapping keys to `TokenIndexer`
            tensors.  At its most basic, using a `SingleIdTokenIndexer` this is : `{"tokens":
            Tensor(batch_size, sequence_length)}`. This dictionary will have the same keys as were used
            for the `TokenIndexers` when you created the `TextField` representing your
            sequence.  The dictionary is designed to be passed directly to a `TextFieldEmbedder`,
            which knows how to combine different word representations into a single vector per
            token in your input.
        metadata : `List[Dict[str, Any]]`, optional (default=`None`)
            A dictionary of metadata for each batch element which has keys:
                words : `List[str]`, required.
                    The tokens in the original sentence.
                pos : `List[str]`, required.
                    The dependencies POS tags for each word.
        head_tags : `torch.LongTensor`, optional (default = `None`)
            A torch tensor representing the sequence of integer gold class labels for the arcs
            in the dependency parse. Has shape `(batch_size, sequence_length)`.
        head_indices : `torch.LongTensor`, optional (default = `None`)
            A torch tensor representing the sequence of integer indices denoting the parent of every
            word in the dependency parse. Has shape `(batch_size, sequence_length)`.

        # Returns

        An output dictionary consisting of:

        loss : `torch.FloatTensor`, optional
            A scalar loss to be optimised.
        arc_loss : `torch.FloatTensor`
            The loss contribution from the unlabeled arcs.
        loss : `torch.FloatTensor`, optional
            The loss contribution from predicting the dependency
            tags for the gold arcs.
        heads : `torch.FloatTensor`
            The predicted head indices for each word. A tensor
            of shape (batch_size, sequence_length).
        head_types : `torch.FloatTensor`
            The predicted head types for each arc. A tensor
            of shape (batch_size, sequence_length).
        mask : `torch.Tensor`
            A mask denoting the padded elements in the batch.
        """
        gold_head_tags = None
        gold_head_indices = None
        if golds != None:
            gold_head_tags=golds['rels']
            gold_head_indices=golds['heads']# TODO, why are they opposite?

        predicted_heads, predicted_head_tags, _, topn_heads_indices, topn_heads_values, topn_labels_indices, \
        topn_labels_values, arc_nll, tag_nll = self._parse(embedded_text, mask, gold_head_tags, gold_head_indices)

        if topn_heads_indices != None:
            topn_heads_indices = topn_heads_indices[:,1:,:]
            topn_heads_values = topn_heads_values[:,1:,:]
            topn_labels_indices = topn_labels_indices[:,1:,:]
            topn_labels_values = topn_labels_values[:,1:,:]

        out_dict = dict(predicted_heads=predicted_heads, predicted_head_tags=predicted_head_tags,
                        topn_heads_indices=topn_heads_indices, topn_heads_values=topn_heads_values,
                        topn_labels_indices=topn_labels_indices, topn_labels_values=topn_labels_values,
                        arc_nll=arc_nll, tag_nll=tag_nll)

        if type(gold_head_tags) != type(None):
            self.metric.score(predicted_heads, predicted_head_tags, gold_head_indices, gold_head_tags)
            if self.additional_metrics:
                for additional_metric in self.additional_metrics:
                    additional_metric.score(predicted_heads, predicted_head_tags, gold_head_indices, gold_head_tags)
                # logger.error('Error, additional_metrics for dependency task type is not supported yet')
            loss = (arc_nll + tag_nll) * self.loss_weight
            out_dict['loss'] = loss
        return out_dict

    def get_output_labels(self, mlm_out, mask, golds):
        head_tag_labels = []
        head_indices = []
        tag_probs = []
        indice_probs = []

        lengths = mask.sum(-1)
        
        forward_dict = self.forward(mlm_out, mask, golds)
        heads = forward_dict['predicted_heads']
        head_tags = forward_dict['predicted_head_tags']
        topn_heads_indices = forward_dict['topn_heads_indices']
        topn_heads_values = forward_dict['topn_heads_values']
        topn_labels_indices = forward_dict['topn_labels_indices']
        topn_labels_values = forward_dict['topn_labels_values']

        if self.topn not in [0, 1, None]:
            for length, topn_heads, topn_head_probs, topn_labels, topn_label_probs in zip(lengths, topn_heads_indices,
                                                                                          topn_heads_values,
                                                                                          topn_labels_indices,
                                                                                          topn_labels_values):
                sent_indices = []
                sent_indice_probs = []
                sent_labels = []
                sent_label_probs = []
                for word_idx in range(0, length):
                    sent_labels.append(
                        [self.vocabulary.id2token(label.item(), self.task) for label in topn_labels[word_idx]])
                    sent_indices.append([str(x.item()) for x in topn_heads[word_idx]])
                    sent_indice_probs.append([x.item() for x in topn_head_probs[word_idx]])
                    sent_label_probs.append([x.item() for x in topn_label_probs[word_idx]])
                head_tag_labels.append(sent_labels)
                head_indices.append(sent_indices)
                indice_probs.append(sent_indice_probs)
                tag_probs.append(sent_label_probs)
            return {'dep_labels': head_tag_labels, 'dep_indices': head_indices, 'indice_probs': indice_probs,
                    'tag_probs': tag_probs}
        else:
            for instance_heads, instance_tags, length in zip(heads, head_tags, lengths):
                instance_heads = [x.item() for x in instance_heads]
                instance_tags = [int(x) for x in instance_tags]
                labels = [
                    self.vocabulary.id2token(label, self.task) for label in instance_tags
                ]
                head_tag_labels.append(labels)
                head_indices.append(instance_heads)
            return {'dep_labels': head_tag_labels, 'dep_indices': head_indices}

    def _parse(
            self,
            encoded_text: torch.LongTensor,
            mask: torch.Tensor,
            head_tags: torch.LongTensor = None,
            head_indices: torch.LongTensor = None,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
        torch.Tensor]:

        batch_size, _, encoding_dim = encoded_text.size()

        head_sentinel = self._head_sentinel.expand(batch_size, 1, encoding_dim)
        # Concatenate the head sentinel onto the sentence representation.
        encoded_text = torch.cat([head_sentinel, encoded_text], 1)
        if self.decoder_dropout.p > 0.0:
            encoded_text = self.decoder_dropout(encoded_text)
            
        mask = torch.cat([mask.new_ones(batch_size, 1), mask], 1)
        if mask.dtype != torch.bool:
            mask = mask > 0

        if head_indices is not None:
            head_indices = torch.cat([head_indices.new_zeros(batch_size, 1), head_indices], 1)
        if head_tags is not None:
            head_tags = torch.cat([head_tags.new_zeros(batch_size, 1), head_tags], 1)

        head_arc_representation = torch.nn.ELU(self.head_arc_feedforward(encoded_text)).alpha
        child_arc_representation = torch.nn.ELU(self.child_arc_feedforward(encoded_text)).alpha

        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag_representation = self.head_tag_feedforward(encoded_text)
        child_tag_representation = self.child_tag_feedforward(encoded_text)

        # shape (batch_size, sequence_length, arc_representation_dim)
        # shape (batch_size, sequence_length, sequence_length)
        attended_arcs = self.arc_attention(head_arc_representation, child_arc_representation)

        minus_inf = -1e8
        minus_mask = ~mask * minus_inf
        attended_arcs = attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        topn_heads_indices = None
        topn_heads_values = None
        topn_labels_indices = None
        topn_labels_values = None
        self.get_top_n = self.topn != 1
        if self.get_top_n:
            predicted_heads, predicted_head_tags, topn_heads_indices, topn_heads_values, topn_labels_indices, \
            topn_labels_values = self._greedy_decode(head_tag_representation, child_tag_representation, attended_arcs,
                                                     mask)
        else:
            predicted_heads, predicted_head_tags = self._mst_decode(
                head_tag_representation, child_tag_representation, attended_arcs, mask
            )
        if head_indices is not None and head_tags is not None:

            arc_nll, tag_nll = self._construct_loss(
                head_tag_representation=head_tag_representation,
                child_tag_representation=child_tag_representation,
                attended_arcs=attended_arcs,
                head_indices=head_indices,
                head_tags=head_tags,
                mask=mask,
            )
        else:
            arc_nll, tag_nll = self._construct_loss(
                head_tag_representation=head_tag_representation,
                child_tag_representation=child_tag_representation,
                attended_arcs=attended_arcs,
                head_indices=predicted_heads.long(),
                head_tags=predicted_head_tags.long(),
                mask=mask,
            )
        return predicted_heads[:, 1:], predicted_head_tags[:, 1:], mask, topn_heads_indices, topn_heads_values, \
               topn_labels_indices, topn_labels_values, arc_nll, tag_nll

    def _construct_loss(
            self,
            head_tag_representation: torch.Tensor,
            child_tag_representation: torch.Tensor,
            attended_arcs: torch.Tensor,
            head_indices: torch.Tensor,
            head_tags: torch.Tensor,
            mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the arc and tag loss for a sequence given gold head indices and tags.

        # Parameters

        head_tag_representation : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.
        head_indices : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length).
            The indices of the heads for every word.
        head_tags : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length).
            The dependency labels of the heads for every word.
        mask : `torch.Tensor`, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.

        # Returns

        arc_nll : `torch.Tensor`, required.
            The negative log likelihood from the arc loss.
        tag_nll : `torch.Tensor`, required.
            The negative log likelihood from the arc tag loss.
        """
        # map mask to special token, to avoid out of bounds
        head_indices[head_indices==-100] = 0
        head_tags[head_tags==-100] = 0

        batch_size, sequence_length, _ = attended_arcs.size()
        # shape (batch_size, 1)
        range_vector = get_range_vector(batch_size, attended_arcs.get_device()).unsqueeze(1)
        # shape (batch_size, sequence_length, sequence_length)
        normalised_arc_logits = (
                masked_log_softmax(attended_arcs, mask) * mask.unsqueeze(2) * mask.unsqueeze(1)
        )

        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(
            head_tag_representation, child_tag_representation, head_indices
        )
        normalised_head_tag_logits = masked_log_softmax(
            head_tag_logits, mask.unsqueeze(-1)
        ) * mask.unsqueeze(-1)
        # index matrix with shape (batch, sequence_length)
        timestep_index = get_range_vector(sequence_length, attended_arcs.get_device())
        child_index = (
            timestep_index.view(1, sequence_length).expand(batch_size, sequence_length).long()
        )
        # shape (batch_size, sequence_length)
        arc_loss = normalised_arc_logits[range_vector, child_index, head_indices]
        tag_loss = normalised_head_tag_logits[range_vector, child_index, head_tags]
        # We don't care about predictions for the symbolic ROOT token's head,
        # so we remove it from the loss.
        arc_loss = arc_loss[:, 1:]
        tag_loss = tag_loss[:, 1:]

        # The number of valid positions is equal to the number of unmasked elements minus
        # 1 per sequence in the batch, to account for the symbolic HEAD token.
        valid_positions = mask.sum() - batch_size

        arc_nll = -arc_loss.sum() / valid_positions.float()
        tag_nll = -tag_loss.sum() / valid_positions.float()
        return arc_nll, tag_nll

    def _greedy_decode(
            self,
            head_tag_representation: torch.Tensor,
            child_tag_representation: torch.Tensor,
            attended_arcs: torch.Tensor,
            mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions by decoding the unlabeled arcs
        independently for each word and then again, predicting the head tags of
        these greedily chosen arcs independently. Note that this method of decoding
        is not guaranteed to produce trees (i.e. there maybe be multiple roots,
        or cycles when children are attached to their parents).

        # Parameters

        head_tag_representation : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.

        # Returns

        heads : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        head_tags : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length) representing the
            dependency tags of the greedily decoded heads of each word.
        """
        # Mask the diagonal, because the head of a word can't be itself.
        attended_arcs = attended_arcs + torch.diag(
            attended_arcs.new(mask.size(1)).fill_(-numpy.inf)
        )
        # Mask padded tokens, because we only want to consider actual words as heads.
        if mask is not None:
            minus_mask = ~mask.unsqueeze(2)
            attended_arcs.masked_fill_(minus_mask, -numpy.inf)

        # Compute the heads greedily.
        # shape (batch_size, sequence_length)
        _, heads = attended_arcs.max(dim=2)

        # Given the greedily predicted heads, decode their dependency tags.
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self._get_head_tags(
            head_tag_representation, child_tag_representation, heads
        )
        _, head_tags = head_tag_logits.max(dim=2)
        if self.get_top_n:
            probs = F.softmax(attended_arcs, -1)
            topn = min(probs.shape[2], self.topn)
            topk_heads = torch.topk(probs, topn, dim=2)
            head_tag_logits = head_tag_logits[:, :, 1:]
            topk_labels = torch.topk(F.softmax(head_tag_logits, -1), self.topn, dim=2)
            topk_labels_indices = torch.add(topk_labels.indices, 1)
            return heads, head_tags, topk_heads.indices, topk_heads.values, topk_labels_indices, topk_labels.values
        else:
            return heads, head_tags, None, None, None, None

    def _mst_decode(
            self,
            head_tag_representation: torch.Tensor,
            child_tag_representation: torch.Tensor,
            attended_arcs: torch.Tensor,
            mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Decodes the head and head tag predictions using the Edmonds' Algorithm
        for finding minimum spanning trees on directed graphs. Nodes in the
        graph are the words in the sentence, and between each pair of nodes,
        there is an edge in each direction, where the weight of the edge corresponds
        to the most likely dependency label probability for that arc. The MST is
        then generated from this directed graph.

        # Parameters

        head_tag_representation : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        attended_arcs : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, sequence_length) used to generate
            a distribution over attachments of a given word to all other words.

        # Returns

        heads : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length) representing the
            greedily decoded heads of each word.
        head_tags : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length) representing the
            dependency tags of the optimally decoded heads of each word.
        """
        batch_size, sequence_length, tag_representation_dim = head_tag_representation.size()

        lengths = mask.data.sum(dim=1).long().cpu().numpy()

        expanded_shape = [batch_size, sequence_length, sequence_length, tag_representation_dim]
        head_tag_representation = head_tag_representation.unsqueeze(2)
        head_tag_representation = head_tag_representation.expand(*expanded_shape).contiguous()
        child_tag_representation = child_tag_representation.unsqueeze(1)
        child_tag_representation = child_tag_representation.expand(*expanded_shape).contiguous()
        # Shape (batch_size, sequence_length, sequence_length, num_head_tags)
        pairwise_head_logits = self.tag_bilinear(head_tag_representation, child_tag_representation)

        # Note that this log_softmax is over the tag dimension, and we don't consider pairs
        # of tags which are invalid (e.g are a pair which includes a padded element) anyway below.
        # Shape (batch, num_labels,sequence_length, sequence_length)
        normalized_pairwise_head_logits = F.log_softmax(pairwise_head_logits, dim=3).permute(
            0, 3, 1, 2
        )

        # Mask padded tokens, because we only want to consider actual words as heads.
        minus_inf = -1e8
        minus_mask = ~mask * minus_inf
        attended_arcs = attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        # Shape (batch_size, sequence_length, sequence_length)
        normalized_arc_logits = F.log_softmax(attended_arcs, dim=2).transpose(1, 2)

        # Shape (batch_size, num_head_tags, sequence_length, sequence_length)
        # This energy tensor expresses the following relation:
        # energy[i,j] = "Score that i is the head of j". In this
        # case, we have heads pointing to their children.
        batch_energy = torch.exp(
            normalized_arc_logits.unsqueeze(1) + normalized_pairwise_head_logits
        )
        return self._run_mst_decoding(batch_energy, lengths)

    @staticmethod
    def _run_mst_decoding(
            batch_energy: torch.Tensor, lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        heads = []
        head_tags = []
        for energy, length in zip(batch_energy.detach().cpu(), lengths):
            scores, tag_ids = energy.max(dim=0)
            # Although we need to include the root node so that the MST includes it,
            # we do not want any word to be the parent of the root node.
            # Here, we enforce this by setting the scores for all word -> ROOT edges
            # edges to be 0.
            scores[0, :] = 0
            # Decode the heads. Because we modify the scores to prevent
            # adding in word -> ROOT edges, we need to find the labels ourselves.
            instance_heads, _ = decode_mst(scores, length, has_labels=False)

            # Find the labels which correspond to the edges in the max spanning tree.
            instance_head_tags = []
            for child, parent in enumerate(instance_heads):
                instance_head_tags.append(tag_ids[parent, child].item())
            # We don't care what the head or tag is for the root token, but by default it's
            # not necessarily the same in the batched vs unbatched case, which is annoying.
            # Here we'll just set them to zero.
            instance_heads[0] = 0
            instance_head_tags[0] = 0
            heads.append(instance_heads)
            head_tags.append(instance_head_tags)
        return (
            torch.from_numpy(numpy.stack(heads)).to(batch_energy.device),
            torch.from_numpy(numpy.stack(head_tags)).to(batch_energy.device),
        )

    def _get_head_tags(
            self,
            head_tag_representation: torch.Tensor,
            child_tag_representation: torch.Tensor,
            head_indices: torch.Tensor,
    ) -> torch.Tensor:
        """
        Decodes the head tags given the head and child tag representations
        and a tensor of head indices to compute tags for. Note that these are
        either gold or predicted heads, depending on whether this function is
        being called to compute the loss, or if it's being called during inference.

        # Parameters

        head_tag_representation : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        child_tag_representation : `torch.Tensor`, required
            A tensor of shape (batch_size, sequence_length, tag_representation_dim),
            which will be used to generate predictions for the dependency tags
            for the given arcs.
        head_indices : `torch.Tensor`, required.
            A tensor of shape (batch_size, sequence_length). The indices of the heads
            for every word.

        # Returns

        head_tag_logits : `torch.Tensor`
            A tensor of shape (batch_size, sequence_length, num_head_tags),
            representing logits for predicting a distribution over tags
            for each arc.
        """
        batch_size = head_tag_representation.size(0)
        # shape (batch_size,)
        range_vector = get_range_vector(batch_size, head_tag_representation.get_device()).unsqueeze(1)

        # This next statement is quite a complex piece of indexing, which you really
        # need to read the docs to understand. See here:
        # https://docs.scipy.org/doc/numpy-1.13.0/reference/arrays.indexing.html#advanced-indexing
        # In effect, we are selecting the indices corresponding to the heads of each word from the
        # sequence length dimension for each element in the batch.
        
        # shape (batch_size, sequence_length, tag_representation_dim)
        selected_head_tag_representations = head_tag_representation[range_vector, head_indices]
        selected_head_tag_representations = selected_head_tag_representations.contiguous()
        # shape (batch_size, sequence_length, num_head_tags)
        head_tag_logits = self.tag_bilinear(
            selected_head_tag_representations, child_tag_representation
        )
        return head_tag_logits
