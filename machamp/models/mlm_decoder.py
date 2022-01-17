from typing import Dict
from overrides import overrides
import random
import copy
import logging
import torch

from allennlp.common.checks import check_dimensions_match
from allennlp.data import TextFieldTensors, Vocabulary
from allennlp.models.model import Model
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.nn import util, InitializerApplicator
from allennlp.training.metrics import Perplexity

from transformers import AutoModelForMaskedLM

from machamp.metrics import Perplexity
logger = logging.getLogger(__name__)

@Model.register("machamp_mlm_decoder")
class MachampMaskedLanguageModel(Model):
    """
    # Parameters

    vocab : `Vocabulary`
    text_field_embedder : `TextFieldEmbedder`
        Used to embed the indexed tokens we get in `forward`.
    language_model_head : `LanguageModelHead`
        The `torch.nn.Module` that goes from the hidden states output by the contextualizer to
        logits over some output vocabulary.
    contextualizer : `Seq2SeqEncoder`, optional (default=`None`)
        Used to "contextualize" the embeddings.  This is optional because the contextualization
        might actually be done in the text field embedder.
    target_namespace : `str`, optional (default=`'bert'`)
        Namespace to use to convert predicted token ids to strings in
        `Model.make_output_human_readable`.
    dropout : `float`, optional (default=`0.0`)
        If specified, dropout is applied to the contextualized embeddings before computation of
        the softmax. The contextualized embeddings themselves are returned without dropout.
    """

    def __init__(
        self,
        task: str,
        vocab: Vocabulary,
        input_dim: int,
        pretrained_model: str,
        loss_weight: float = 1.0,
        dec_dataset_embeds_dim: int = 0,
        metric: str = 'perplexity',
        **kwargs,
    ) -> None:
        super().__init__(vocab, **kwargs)

        # isnt this costly?, why not reuse our encoder?        
        mlm = AutoModelForMaskedLM.from_pretrained(pretrained_model)

        # R: This is somewhat (or very) ugly code, however not sure how to 
        # do this cleaner while supporting so many *ForMaskedLM models
        #
        # ps. I guess distilbert is missing, wasnt sure which to pick
        self.lm_config = mlm.config
        try:
            self.mlm = mlm.pred_layer
        except:
            try:
                self.mlm = mlm.cls
            except:
                try:
                    self.mlm = mlm.lm_head
                except:
                    try:
                        self.mlm = mlm.generator_lm_head
                    except:
                        try:
                            self.mlm = mlm.predictions
                        except:
                            logger.error(pretrained_model + ' not yet configured for masked language modeling')
                            exit(1)

        self.task = task
        self.input_dim = input_dim + dec_dataset_embeds_dim
        self.loss_weight = loss_weight
        self.loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
        
        self.metrics = {
            "perplexity": Perplexity(),
        }

    def forward(  # type: ignore
        self,
        embedded_text: torch.LongTensor,
        gold_labels: torch.LongTensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        # Parameters

        tokens : `TextFieldTensors`
            The output of `TextField.as_tensor()` for a batch of sentences.
        mask_positions : `torch.LongTensor`
            The positions in `tokens` that correspond to [MASK] tokens that we should try to fill
            in.  Shape should be (batch_size, num_masks).
        target_ids : `TextFieldTensors`
            This is a list of token ids that correspond to the mask positions we're trying to fill.
            It is the output of a `TextField`, purely for convenience, so we can handle wordpiece
            tokenizers and such without having to do crazy things in the dataset reader.  We assume
            that there is exactly one entry in the dictionary, and that it has a shape identical to
            `mask_positions` - one target token per mask position.
        """
        logits = self.mlm(embedded_text)
        if type(logits) == tuple:
            logits = logits[0]

        masked_lm_loss = self.loss_fct(logits.view(-1, self.lm_config.vocab_size), gold_labels.view(-1))
        probs = torch.nn.functional.softmax(logits, dim=-1)
        output_dict = {"logits": logits, "class_probabilities": probs}

        output_dict["loss"] = self.loss_weight * masked_lm_loss

        for metric in self.metrics.values():
            metric(masked_lm_loss)

        return output_dict


    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        main_metrics = {
            f".run/{self.task}/{metric_name}": metric.get_metric(reset)
            for metric_name, metric in self.metrics.items()
        }
        return {**main_metrics}

    @overrides
    def make_output_human_readable(
        self, output_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        top_words = []
        for instance_indices in output_dict["top_indices"]:
            top_words.append(
                [
                    [
                        self.vocab.get_token_from_index(
                            index.item(), namespace=self._target_namespace
                        )
                        for index in mask_positions
                    ]
                    for mask_positions in instance_indices
                ]
            )
        output_dict["words"] = top_words
        tokens = []
        for instance_tokens in output_dict["token_ids"]:
            tokens.append(
                [
                    self.vocab.get_token_from_index(
                        token_id.item(), namespace=self._target_namespace
                    )
                    for token_id in instance_tokens
                ]
            )
        output_dict["tokens"] = tokens

        return output_dict

