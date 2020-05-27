"""
The base UDify model for training and prediction
"""

from typing import Optional, Any, Dict, List, Tuple
from overrides import overrides
import logging

import torch

from pytorch_pretrained_bert.tokenization import BertTokenizer

from allennlp.common.checks import check_dimensions_match, ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import Seq2SeqEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn.util import get_text_field_mask

from machamp.modules.scalar_mix import ScalarMixWithDropout

import machamp.dataset_readers.universal_reader as reader

logger = logging.getLogger(__name__)


@Model.register("machamp_model")
class MachampModel(Model):
    """
    The UDify model base class. Applies a sequence of shared encoders before decoding in a multi-task configuration.
    Uses TagDecoder and DependencyDecoder to decode each UD task.
    """

    def __init__(self,
                 vocab: Vocabulary,
                 tasks: List[str],
                 task_types: List[str],
                 text_field_embedder: TextFieldEmbedder,
                 encoder: Seq2SeqEncoder,
                 decoders: Dict[str, Model],
                 layers_for_tasks: list = [int],
                 weight_embeddings: bool = True,
                 post_encoder_embedder: TextFieldEmbedder = None,
                 dropout: float = 0.0,
                 word_dropout: float = 0.0,
                 mix_embedding: bool = False,
                 layer_dropout: int = 0.0,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 bert_path: str = "",
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super(MachampModel, self).__init__(vocab, regularizer)

        # A: task list will be used for the order !!!
        self.tasks = tasks
        self.task_types = task_types
        self.vocab = vocab
        self.bert_vocab = BertTokenizer.from_pretrained(bert_path).vocab
        self.text_field_embedder = text_field_embedder
        self.post_encoder_embedder = post_encoder_embedder
        self.shared_encoder = encoder
        self.word_dropout = word_dropout
        self.dropout = torch.nn.Dropout(p=dropout)
        self.decoders = torch.nn.ModuleDict(decoders)

        # A: to pass previous task embeddings with weighted distribution
        self.weight_embeddings = weight_embeddings

        # A: BERT layers to pass output of those layers to tasks separately
        self.layers_for_tasks = layers_for_tasks

        if mix_embedding:
            self.scalar_mix = torch.nn.ModuleDict({
                task: ScalarMixWithDropout(self.layers_for_tasks[self.tasks.index(task)],
                                           do_layer_norm=False,
                                           dropout=layer_dropout)
                for task in self.tasks
            })
        else:
            self.scalar_mix = None

        self.metrics = {}

        for task in self.tasks:
            if task not in self.decoders:
                raise ConfigurationError(f"Task {task} has no corresponding decoder. Make sure their names match.")

        check_dimensions_match(text_field_embedder.get_output_dim(), encoder.get_input_dim(),
                               "text field embedding dim", "encoder input dim")

        initializer(self)
        self._count_params()

    @overrides
    def forward(self,
                tokens: Dict[str, torch.LongTensor],
                metadata: List[Dict[str, Any]] = None,
                **kwargs: Dict[str, torch.LongTensor]) -> Dict[str, torch.Tensor]:
        
        if "track_epoch" in kwargs:
            track_epoch = kwargs.pop("track_epoch")

        gold_tags = kwargs

        if "tokens" in self.tasks:
            # Model is predicting tokens, so add them to the gold tags
            gold_tags["tokens"] = tokens["tokens"]

        mask = get_text_field_mask(tokens)
        self._apply_token_dropout(tokens)

        # flag to indicate the task is token level or not (for current batch)
        is_word_level = [a['isWordLevel'][0] for a in metadata]
        assert len(set(is_word_level)) == 1, 'a batch has not unique value'
        is_word_level = is_word_level[0]

        embedded_text_input = self.text_field_embedder(tokens)

        if self.post_encoder_embedder:
            post_embeddings = self.post_encoder_embedder(tokens)

        encoded_text = self.shared_encoder(embedded_text_input, mask) if is_word_level else self.shared_encoder(embedded_text_input)

        logits = {}
        class_probabilities = {}
        output_dict = {"logits": logits,
                       "class_probabilities": class_probabilities}
        loss = 0

        # Run through each of the tasks on the shared encoder and save predictions

        # A: keep previous task to pass the predicted logit to the next task
        prev_task = self.tasks[0]
        pred_classes = None
        for i in range(len(self.tasks)):
            task = self.tasks[i]
            #TODO, this is disabling both tag-decoder/sent-decoder in one batch
            checkTask = task
            if self.task_types[i] == 'dependency':
                checkTask = 'head_tags'
            #TODO fix nicer, this is hardcoded when only dataset, isWordLevel and words are defined
            isRaw = len(gold_tags) == 3
            if checkTask not in gold_tags and not isRaw:
                continue
            if (self.task_types[i] == 'classification') == is_word_level and not isRaw:
                #current task sentence level == current batch is wordlevel 
                continue


            if self.scalar_mix and is_word_level:
                decoder_input = self.scalar_mix[task](encoded_text[:self.layers_for_tasks[i]], mask)
            elif is_word_level:
                decoder_input = encoded_text[self.layers_for_tasks[i]-1]
            else:
                decoder_input = encoded_text

            if self.post_encoder_embedder:
                decoder_input = decoder_input + post_embeddings

            # tag_logits = logits[prev_task] if prev_task in logits else None
            if self.task_types[i] == "dependency":
                # tag_logits = logits["upos"] if "upos" in logits else None
                
                # instead of pos_logit, pass the pre_task logit
                pred_output = self.decoders[task](decoder_input, mask, pred_classes,
                                                  gold_tags.get("head_tags", None), gold_tags.get("head_indices", None), metadata)
                for key in ["heads", "head_tags", "arc_loss", "tag_loss", "mask"]:
                    output_dict[key] = pred_output[key]
                #tmp_tensor = torch.zeros((decoder_input.shape[0], decoder_input.shape[1]))
                #for sent in range(tmp_tensor.shape[0]):
                #    for idx in range(tmp_tensor.shape[1]):
                #        if pred_output['mask'][sent][idx].item() == 1:
                #            #TODO, get rid of this?
                #            #encoded_dep = reader.dep_encoding(idx, output_dict['heads'][sent][idx].item(),
                #                                            self.vocab.get_token_from_index(output_dict['head_tags'][sent][idx].item(), 'head_tags'))
                #            encoded_dep_id = self.vocab.get_token_index(encoded_dep,'dep_encoded')
                #            tmp_tensor[sent][idx] = encoded_dep_id
                #        else:
                #            tmp_tensor[sent][idx] = 0
                #pred_classes = tuple([tmp_tensor.long().to(decoder_input.device), False])
                pred_classes = tuple([pred_output, self.weight_embeddings])


            else:
                pred_output = self.decoders[task](decoder_input, mask, gold_tags, pred_classes, metadata)
                logits[task] = pred_output["logits"]
                class_probabilities[task] = pred_output["class_probabilities"]
                if self.weight_embeddings:
                    pred_classes = tuple([class_probabilities[task], self.weight_embeddings])
                else:
                    pred_classes = tuple([class_probabilities[task].max(-1)[1], self.weight_embeddings])

            if task in gold_tags or self.task_types[i] == "dependency" and "head_tags" in gold_tags:
                # Keep track of the loss if we have the gold tags available
                loss += pred_output["loss"]

            if self.task_types[i] != 'classification':
                prev_task = task

        if gold_tags:
            output_dict["loss"] = loss

        if metadata is not None:
            output_dict["words"] = [x["words"] for x in metadata]
            output_dict["fullData"] = [x['fullData'] for x in metadata]
            output_dict["copy_other_columns"] = [x['copy_other_columns'] for x in metadata]
            output_dict["colIdxs"] = [x['colIdxs'] for x in metadata]
            
            #Rob: Warning, hacky!, allennlp requires them to be in the length of metadata, in the dump_lines I just use the first
            output_dict['tasks'] = [self.tasks for x in metadata]
            output_dict["task_types"] = [self.task_types for x in metadata]
        return output_dict

    def _apply_token_dropout(self, tokens):
        # Word dropout
        if "tokens" in tokens:
            oov_token = self.vocab.get_token_index(self.vocab._oov_token)
            ignore_tokens = [self.vocab.get_token_index(self.vocab._padding_token)]
            tokens["tokens"] = self.token_dropout(tokens["tokens"],
                                                  oov_token=oov_token,
                                                  padding_tokens=ignore_tokens,
                                                  p=self.word_dropout,
                                                  training=self.training)

        # BERT token dropout
        if "bert" in tokens:
            oov_token = self.bert_vocab["[MASK]"]
            ignore_tokens = [self.bert_vocab["[PAD]"], self.bert_vocab["[CLS]"], self.bert_vocab["[SEP]"]]
            tokens["bert"] = self.token_dropout(tokens["bert"],
                                                oov_token=oov_token,
                                                padding_tokens=ignore_tokens,
                                                p=self.word_dropout,
                                                training=self.training)

    @staticmethod
    def token_dropout(tokens: torch.LongTensor,
                      oov_token: int,
                      padding_tokens: List[int],
                      p: float = 0.2,
                      training: float = True) -> torch.LongTensor:
        """
        During training, randomly replaces some of the non-padding tokens to a mask token with probability ``p``

        :param tokens: The current batch of padded sentences with word ids
        :param oov_token: The mask token
        :param padding_tokens: The tokens for padding the input batch
        :param p: The probability a word gets mapped to the unknown token
        :param training: Applies the dropout if set to ``True``
        :return: A copy of the input batch with token dropout applied
        """
        if training and p > 0:
            # Ensure that the tensors run on the same device
            device = tokens.device

            # This creates a mask that only considers unpadded tokens for mapping to oov
            padding_mask = torch.ones(tokens.size(), dtype=torch.bool).to(device)
            for pad in padding_tokens:
                padding_mask &= (tokens != pad)

            # Create a uniformly random mask selecting either the original words or OOV tokens
            dropout_mask = (torch.empty(tokens.size()).uniform_() < p).to(device)
            oov_mask = dropout_mask & padding_mask

            oov_fill = torch.empty(tokens.size(), dtype=torch.long).fill_(oov_token).to(device)

            result = torch.where(oov_mask, oov_fill, tokens)

            return result
        else:
            return tokens

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        for task in self.tasks:
            isDep = self.task_types[self.tasks.index(task)] == 'dependency'
            if task in output_dict['class_probabilities'] or (isDep and 'heads' in output_dict):
                self.decoders[task].decode(output_dict)
        if output_dict['loss'] == 0:
            output_dict['loss'] = [output_dict['loss']]
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        for task in self.tasks:
            for name, task_metric in self.decoders[task].get_metrics(reset).items():
                if name.split("/")[-1] in ["acc", 'UAS', 'LAS', 'UEM', 'LEM']:
                    metrics[name] = task_metric
                elif name.split('/')[-1] in ['micro-f1', 'macro-f1']:
                    metrics[name] = task_metric['fscore']
                elif name.split("/")[-1] == "span_f1":
                    metrics[name] = task_metric["f1-measure-overall"]
                    for subname, submetric in task_metric.items():
                        logger.info(f"{subname} {submetric}.")
                elif name.split("/")[-1] == "multi_span_f1":
                    metrics[name] = task_metric["f1-measure-overall"]
                    for subname, submetric in task_metric.items():
                        logger.info(f"{subname} {submetric}.")
                else:
                    logger.warning(f"ERROR. Metric: {name} unrecognized.")
                    import sys
                    sys.exit()

        # The "sum" metric summing all tracked metrics keeps a good measure of patience for early stopping and saving
        metrics_to_track = set()
        for task, task_type in zip(self.tasks, self.task_types):
            metrics_to_track.add(task if task_type != 'dependency' else 'LAS')

        metrics[".run/.sum"] = sum(metric
                                   for name, metric in metrics.items()
                                   if not name.startswith("_") and set(name.split("/")).intersection(metrics_to_track))

        return metrics

    def _count_params(self):
        self.total_params = sum(p.numel() for p in self.parameters())
        self.total_train_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        logger.info(f"Total number of parameters: {self.total_params}")
        logger.info(f"Total number of trainable parameters: {self.total_train_params}")
