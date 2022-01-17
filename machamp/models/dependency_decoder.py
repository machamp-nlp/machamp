import copy
import logging
from typing import Dict, Tuple

import numpy
import torch
import torch.nn.functional as F
from allennlp.common.checks import check_dimensions_match
from allennlp.data import Vocabulary
from allennlp.models.model import Model
from allennlp.modules import FeedForward
from allennlp.modules.matrix_attention.bilinear_matrix_attention import BilinearMatrixAttention
from allennlp.nn import Activation
from allennlp.nn.chu_liu_edmonds import decode_mst
from allennlp.nn.util import get_device_of, masked_log_softmax, get_lengths_from_binary_sequence_mask
from allennlp.nn.util import get_range_vector
from allennlp.training.metrics import AttachmentScores
from overrides import overrides

logger = logging.getLogger(__name__)


@Model.register("machamp_dependency_decoder")
class MachampBiaffineDependencyParser(Model):
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

    vocab : `Vocabulary`, required
        A Vocabulary, required in order to compute sizes for input/output projections.
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
    use_mst_decoding_for_validation : `bool`, optional (default = `True`).
        Whether to use Edmond's algorithm to find the optimal minimum spanning tree during validation.
        If false, decoding is greedy.
    dropout : `float`, optional, (default = `0.0`)
        The variational dropout applied to the output of the encoder and MLP layers.
    """

    def __init__(
        self,
        task: str,
        vocab: Vocabulary,
        input_dim: int,
        tag_representation_dim: int,
        arc_representation_dim: int,
        loss_weight: float=1.0,
        metric: str = 'las',
        use_mst_decoding_for_validation: bool = True,
        dec_dataset_embeds_dim: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        self.task = task
        self.input_dim = input_dim + dec_dataset_embeds_dim
        arc_representation_dim = arc_representation_dim + dec_dataset_embeds_dim 
        self.loss_weight = loss_weight

        self.head_arc_feedforward = FeedForward(
            self.input_dim, 1, arc_representation_dim, Activation.by_name("elu")()
        )
        self.child_arc_feedforward = copy.deepcopy(self.head_arc_feedforward)

        self.arc_attention = BilinearMatrixAttention(
            arc_representation_dim, arc_representation_dim, use_input_biases=True
        )

        num_labels = self.vocab.get_vocab_size(task + "_rels")

        self.head_tag_feedforward = FeedForward(
            self.input_dim, 1, tag_representation_dim, Activation.by_name("elu")()
        )
        self.child_tag_feedforward = copy.deepcopy(self.head_tag_feedforward)

        self.tag_bilinear = torch.nn.modules.Bilinear(
            tag_representation_dim, tag_representation_dim, num_labels
        )

        self._head_sentinel = torch.nn.Parameter(torch.randn([1, 1, self.input_dim]))

        representation_dim = self.input_dim#text_field_embedder.get_output_dim()

        check_dimensions_match(
            representation_dim,
            self.input_dim,
            "text field embedding dim",
            "encoder input dim",
        )

        check_dimensions_match(
            tag_representation_dim,
            self.head_tag_feedforward.get_output_dim(),
            "tag representation dim",
            "tag feedforward output dim",
        )
        check_dimensions_match(
            arc_representation_dim,
            self.head_arc_feedforward.get_output_dim(),
            "arc representation dim",
            "arc feedforward output dim",
        )

        self.use_mst_decoding_for_validation = use_mst_decoding_for_validation

        self.metrics = {
            "las": AttachmentScores(),
        }



    @overrides
    def forward(
        self,  # type: ignore
        embedded_text: torch.LongTensor,
        mask: torch.LongTensor = None,
        gold_head_tags: torch.LongTensor = None,
        gold_head_indices: torch.LongTensor = None,
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
        mask : `torch.BoolTensor`
            A mask denoting the padded elements in the batch.
        """

        predicted_heads, predicted_head_tags, mask, arc_nll, tag_nll = self._parse(
            embedded_text, mask, gold_head_tags, gold_head_indices
        )

        loss = (arc_nll + tag_nll) * self.loss_weight

        if gold_head_indices is not None and gold_head_tags is not None:
            # We calculate attachment scores for the whole sentence
            # but excluding the symbolic ROOT token at the start,
            # which is why we start from the second element in the sequence.
            self.metrics['las'](
                predicted_heads[:, 1:],
                predicted_head_tags[:, 1:],
                gold_head_indices,
                gold_head_tags,
                mask[:,1:],
            )

        output_dict = {
            self.task + "_rels": predicted_head_tags,
            self.task + "_head_indices": predicted_heads,
            "loss": loss}
        return output_dict

    @overrides
    def make_output_human_readable(
        self,
        label_predictions: torch.Tensor,
        head_predictions: torch.Tensor,
        mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

        head_tags = label_predictions
        heads = head_predictions
        lengths = get_lengths_from_binary_sequence_mask(mask) 
        head_tag_labels = []
        head_indices = []
        for instance_heads, instance_tags, length in zip(heads, head_tags, lengths):
            instance_heads = list(instance_heads[1:length+1])
            instance_tags = [int(x) for x in instance_tags[1:length+1]]
            labels = [
                self.vocab.get_token_from_index(label, self.task + "_rels") for label in instance_tags
            ]
            # TODO, not sure why multiple roots are predicted, but its hard to find 
            # where this happens. It is very rare, so for now we use a rule-based fix
            if [int(x) for x in instance_heads].count(0) > 1:
                root = -1
                for idx, head in enumerate(instance_heads):
                    if int(head) == 0:
                        if root == -1:
                            root = idx
                        else:
                            instance_heads[idx] = root + 1
                            labels[idx] = "parataxis"
            head_tag_labels.append(labels)
            head_indices.append(instance_heads)
            
        return head_tag_labels, head_indices


    def _parse(
        self,
        encoded_text: torch.LongTensor,
        mask: torch.BoolTensor,
        head_tags: torch.LongTensor = None,
        head_indices: torch.LongTensor = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size, _, encoding_dim = encoded_text.size()

        head_sentinel = self._head_sentinel.expand(batch_size, 1, encoding_dim)
        # Concatenate the head sentinel onto the sentence representation.
        encoded_text = torch.cat([head_sentinel, encoded_text], 1)
        mask = torch.cat([mask.new_ones(batch_size, 1), mask], 1)
        if head_indices is not None:
            head_indices = torch.cat([head_indices.new_zeros(batch_size, 1), head_indices], 1)
        if head_tags is not None:
            head_tags = torch.cat([head_tags.new_zeros(batch_size, 1), head_tags], 1)

        head_arc_representation = self.head_arc_feedforward(encoded_text)
        child_arc_representation = self.child_arc_feedforward(encoded_text)

        # shape (batch_size, sequence_length, tag_representation_dim)
        head_tag_representation = self.head_tag_feedforward(encoded_text)
        child_tag_representation = self.child_tag_feedforward(encoded_text)

        # shape (batch_size, sequence_length, arc_representation_dim)
        # shape (batch_size, sequence_length, sequence_length)
        attended_arcs = self.arc_attention(head_arc_representation, child_arc_representation)

        minus_inf = -1e8
        minus_mask = ~mask * minus_inf
        attended_arcs = attended_arcs + minus_mask.unsqueeze(2) + minus_mask.unsqueeze(1)

        if self.training or not self.use_mst_decoding_for_validation:
            predicted_heads, predicted_head_tags = self._greedy_decode(
                head_tag_representation, child_tag_representation, attended_arcs, mask
            )
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
        return predicted_heads, predicted_head_tags, mask, arc_nll, tag_nll

    def _construct_loss(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        attended_arcs: torch.Tensor,
        head_indices: torch.Tensor,
        head_tags: torch.Tensor,
        mask: torch.BoolTensor,
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
        mask : `torch.BoolTensor`, required.
            A mask of shape (batch_size, sequence_length), denoting unpadded
            elements in the sequence.

        # Returns

        arc_nll : `torch.Tensor`, required.
            The negative log likelihood from the arc loss.
        tag_nll : `torch.Tensor`, required.
            The negative log likelihood from the arc tag loss.
        """
        batch_size, sequence_length, _ = attended_arcs.size()
        # shape (batch_size, 1)
        range_vector = get_range_vector(batch_size, get_device_of(attended_arcs)).unsqueeze(1)
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
        timestep_index = get_range_vector(sequence_length, get_device_of(attended_arcs))
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
        mask: torch.BoolTensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        return heads, head_tags

    def _mst_decode(
        self,
        head_tag_representation: torch.Tensor,
        child_tag_representation: torch.Tensor,
        attended_arcs: torch.Tensor,
        mask: torch.BoolTensor,
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
            instance_heads, _ = decode_mst(scores.numpy(), length, has_labels=False)

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
        range_vector = get_range_vector(
            batch_size, get_device_of(head_tag_representation)
        ).unsqueeze(1)

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

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        main_metrics = {
            f".run/{self.task}/{metric_name}": metric.get_metric(reset)
            for metric_name, metric in self.metrics.items()
        }
        return {**main_metrics}

