import logging
from typing import Dict, Union, Sequence, Set, Optional, cast, Iterator, List

import torch
from allennlp.common.checks import ConfigurationError
from allennlp.data.fields.field import Field
from allennlp.data.fields.sequence_field import SequenceField
from allennlp.data.vocabulary import Vocabulary
from overrides import overrides

logger = logging.getLogger(__name__)


class SequenceMultiLabelField(Field[torch.Tensor]):
    """
    Based on SequenceLabelField, but allows for storing multiple labels per index
    of the sequence.
    """

    # It is possible that users want to use this field with a namespace which uses OOV/PAD tokens.
    # This warning will be repeated for every instantiation of this class (i.e for every data
    # instance), spewing a lot of warnings so this class variable is used to only log a single
    # warning per namespace.
    _already_warned_namespaces: Set[str] = set()

    def __init__(
        self,
        labels: Sequence[Sequence[Union[str, int]]],
        sequence_field: SequenceField,
        label_namespace: str = "labels",
        skip_indexing: bool = False,
        num_labels: Optional[int] = None,
    ) -> None:
        self.labels = labels
        self.sequence_field = sequence_field
        self._label_namespace = label_namespace
        self._indexed_labels = None
        self._label_ids = None
        self._maybe_warn_for_namespace(label_namespace)
        self._num_labels = num_labels

        if len(labels) != sequence_field.sequence_length():
            raise ConfigurationError(
                "Label length and sequence length "
                "don't match: %d and %d" % (len(labels), sequence_field.sequence_length())
            )

        self._skip_indexing = False
        for label_list in labels:
            if all(isinstance(x, int) for x in label_list):
                self._indexed_labels = labels
                self._skip_indexing = True

            elif not all(isinstance(x, str) for x in label_list):
                raise ConfigurationError(
                    "SequenceLabelFields must be passed either all "
                    "strings or all ints. Found labels {} with "
                    "types: {}.".format(label_list, [type(x) for x in label_list])
                )

        if skip_indexing and self.labels:
            if not all(isinstance(label, int) for label in labels):
                raise ConfigurationError(
                    "In order to skip indexing, your labels must be integers. "
                    "Found labels = {}".format(labels)
                )
            if not num_labels:
                raise ConfigurationError("In order to skip indexing, num_labels can't be None.")

            if not all(cast(int, label) < num_labels for label in labels):
                raise ConfigurationError(
                    "All labels should be < num_labels. "
                    "Found num_labels = {} and labels = {} ".format(num_labels, labels)
                )

            self._label_ids = labels
        else:
            for label_list in labels:
                if not all(isinstance(label, str) for label in label_list):
                    raise ConfigurationError(
                        "SequenceMultiLabelFields expects string labels if skip_indexing=False. "
                        "Found labels: {}".format(labels)
                    )

    def _maybe_warn_for_namespace(self, label_namespace: str) -> None:
        if not (label_namespace.endswith("labels") or label_namespace.endswith("tags")):
            if label_namespace not in self._already_warned_namespaces:
                logger.warning(
                    "Your label namespace was '%s'. We recommend you use a namespace "
                    "ending with 'labels' or 'tags', so we don't add UNK and PAD tokens by "
                    "default to your vocabulary.  See documentation for "
                    "`non_padded_namespaces` parameter in Vocabulary.",
                    self._label_namespace,
                )
                self._already_warned_namespaces.add(label_namespace)

    # Sequence methods
    def __iter__(self) -> Iterator[Union[str, int]]:
        return iter(self.labels)

    def __getitem__(self, idx: int) -> Union[str, int]:
        return self.labels[idx]

    def __len__(self) -> int:
        return len(self.labels)

    @overrides
    def count_vocab_items(self, counter: Dict[str, Dict[str, int]]):
        if self._indexed_labels is None:
            for label_list in self.labels:
                for label in label_list:
                    counter[self._label_namespace][label] += 1  # type: ignore

    @overrides
    def index(self, vocab: Vocabulary):
        if not self._skip_indexing:
            self._indexed_labels = []
            for label_list in self.labels:
                token_labels = []
                for label in label_list:
                    token_labels.append(vocab.get_token_index(label, self._label_namespace))
                self._indexed_labels.append(token_labels)

        if not self._num_labels:
            self._num_labels = vocab.get_vocab_size(self._label_namespace)

    @overrides
    def get_padding_lengths(self) -> Dict[str, int]:
        return {"num_tokens": self.sequence_field.sequence_length()}

    @overrides
    def as_tensor(self, padding_lengths: Dict[str, int]) -> torch.Tensor:
        desired_num_tokens = padding_lengths["num_tokens"]
        token_tags_list = self.pad_sequence_to_length_with_list(self._indexed_labels, desired_num_tokens, [0])
        # [[0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [0], [2, 3], [0], [1], [0], [0], [1], [0], [0], [0]]

        label_sequence = []
        for token_index in range(len(token_tags_list)):
            label_vector = [0] * self._num_labels

            # If we are not in padding space, modify the label_vector
            if token_index < len(self._indexed_labels):
                label_indices = self._indexed_labels[token_index]
                # Set the index to 1 if that label is present
                for label_index in label_indices:
                    label_vector[label_index] = 1

            label_sequence.append(label_vector)

        tensor = torch.LongTensor(label_sequence)
        # [[1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 0, 1, 1], [1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0], [1, 0, 0, 0]]

        return tensor


    def pad_sequence_to_length_with_list(
        self,
        sequence: List,
        desired_length: int,
        default_value: List,
        padding_on_right: bool = True,
    ) -> List:
        # Truncates the sequence to the desired length.
        if padding_on_right:
            padded_sequence = sequence[:desired_length]
        else:
            padded_sequence = sequence[-desired_length:]
        # Continues to pad with default_value() until we reach the desired length.
        for _ in range(desired_length - len(padded_sequence)):
            if padding_on_right:
                padded_sequence.append(default_value)
            else:
                padded_sequence.insert(0, default_value)
        return padded_sequence

    @overrides
    def empty_field(self):
        return SequenceMultiLabelField(
            [], self._label_namespace, skip_indexing=True, num_labels=self._num_labels
        )

    def __str__(self) -> str:
        return (
            f"SequenceMultiLabelField with labels: {self.labels} in namespace: '{self._label_namespace}'.'"
        )
