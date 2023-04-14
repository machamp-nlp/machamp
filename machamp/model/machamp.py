import logging
from typing import List, Dict

import torch

logger = logging.getLogger(__name__)

from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM
from transformers import logging as tf_logging

tf_logging.set_verbosity_error()

from machamp.metrics.avg_dist import AvgDist
from machamp.metrics.perplexity import Perplexity
from machamp.data.machamp_vocabulary import MachampVocabulary
from machamp.model.classification_decoder import MachampClassificationDecoder
from machamp.model.regression_decoder import MachampRegressionDecoder
from machamp.model.seq_label_decoder import MachampSeqDecoder
from machamp.model.multiseq_decoder import MachampMultiseqDecoder
from machamp.model.crf_label_decoder import MachampCRFDecoder
from machamp.model.dependency_decoder import MachampDepDecoder
from machamp.model.mlm_decoder import MachampLMDecoder
from machamp.model.multiclas_decoder import MachampMulticlasDecoder
from machamp.model.encoder import MachampEncoder
from machamp.modules.allennlp.scalar_mix import ScalarMix
from machamp.utils import myutils


class MachampModel(torch.nn.Module):
    def __init__(self,
                 vocabulary: MachampVocabulary,
                 tasks: List[str],
                 task_types: List[str],
                 mlm: str,
                 device: str,
                 dataset_configs: Dict,
                 tokenizer: AutoTokenizer,
                 update_weights_encoder: bool,
                 max_input_length: int,
                 retrain: str = '',
                 dropout: float = None,
                 ) -> None:
        """
        The core MaChAmp model, which is basically a wrapper around a 
        transformers.AutoModel and a dictionary of decoders: one for 
        each task.
    
        Parameters
        ----------
        vocabulary: Machamp.data.MachampVocabulary
            Keeps the vocabularies for the output spaces of all tasks
        tasks: list[str]
            List of the names of all tasks, these are the names as defined
            in the dataset configuration
        task_types: list[str]
            List of all task_types, indexed correspondingly to "tasks"
        mlm: str
            Name of the transformers language model to use, can be found on:
            https://huggingface.co/models
        device: str
            Description of cuda device to use, i.e.: "cpu" or "gpu:0"
        dataset_configs: Dict
            The full configuration of all datasets to handle. Included
            here so that we can initialize the decoders correctly
        update_weights_encoder: bool
            Whether we want to update the encoder (the language model), 
            or freeze it.
        max_input_length: int
            The maximum input length to use for the transformers model, 
            this has a huge impact on GPU ram usage.
        retrain: str, 
            Train from a MaChAmp model instead of a transformers model.
            This should have the path to the exact model.
        dropout: float
            Dropout to be applied after the encoder (language model).
        """
        super().__init__()

        # if retrain is specified, we load the MLM from a machamp model
        if retrain not in [None, '']:
            self.mlm = torch.load(retrain).mlm
        # it could be cleaner code if we always use a normal AutoModel, 
        # and load only the prediction MLM layer in mlm_decoder. However, 
        # that would require adaptation for any future model types.
        elif 'mlm' in task_types:
            self.mlm = AutoModelForMaskedLM.from_pretrained(mlm)
        else:
            self.mlm = AutoModel.from_pretrained(mlm)

        if not update_weights_encoder:
            for param in self.mlm.base_model.parameters():
                param.requires_grad = False

        self.mlm.to(device)
        self.vocabulary = vocabulary
        self.tasks = tasks
        self.task_types = task_types
        self.device = device
        self.dataset_configs = dataset_configs

        # Find the size of the masked language model
        if hasattr(self.mlm.config, 'hidden_size'):
            self.mlm_out_size = self.mlm.config.hidden_size
        elif hasattr(self.mlm.config, 'dim'):
            self.mlm_out_size = self.mlm.config.dim
        else:  # if not found, guess
            self.mlm_out_size = 768

        if dropout == None:
            self.dropout = None
        else:
            self.dropout = torch.nn.Dropout(dropout)

        tokenizer_out = tokenizer.prepare_for_model([])['input_ids']
        # we assume that if there is only one special token that it is the end token
        self.end_token = None if len(tokenizer_out) == 0 else tokenizer_out[-1]
        self.start_token = None if len(tokenizer_out) <= 1 else tokenizer_out[0]
        self.num_special_tokens = 2 - [self.end_token, self.start_token].count(None)
        self.encoder = MachampEncoder(self.mlm, max_input_length, self.end_token, self.start_token)

        self.decoders = torch.nn.ModuleDict()
        self.scalars = torch.nn.ModuleDict()
        self.layers = {}
        for task, task_type in zip(self.tasks, self.task_types):

            # should the scalars be done per uniq task/dataset combo?
            # now they are done once if the tasks have the same name
            for dataset in dataset_configs:
                if task in dataset_configs[dataset]['tasks']:
                    break
                if task in dataset_configs[dataset]['tasks']  and 'layers_to_use' not in dataset_configs[dataset]['tasks'][task]:
                    dataset_configs[dataset]['tasks'][task]['layers_to_use'] = [-1]  # not a great place for defaults
                    # but the default is set in params.json, this is just for backwards compatability
            # we pick the number of layers from the last dataset for this task!
            num_layers = len(dataset_configs[dataset]['tasks'][task]['layers_to_use'])
            if num_layers > 1:
                self.scalars[task] = ScalarMix(num_layers)
            else:
                self.scalars[task] = None
            self.layers[task] = dataset_configs[dataset]['tasks'][task]['layers_to_use']

            if task_type == 'classification':
                decoder_type = MachampClassificationDecoder
            elif task_type in ['seq', 'string2string', 'tok']:
                decoder_type = MachampSeqDecoder
            elif task_type == 'seq_bio':
                decoder_type = MachampCRFDecoder
            elif task_type == 'dependency':
                decoder_type = MachampDepDecoder
            elif task_type == 'regression':
                decoder_type = MachampRegressionDecoder
            elif task_type == 'mlm':
                decoder_type = MachampLMDecoder
            elif task_type == 'multiseq':
                decoder_type = MachampMultiseqDecoder
            elif task_type == 'multiclas':
                decoder_type = MachampMulticlasDecoder
            else:
                logger.error('Error, task_type ' + task_type + ' not implemented')
                exit(1)

            decoder = decoder_type(task, self.vocabulary, self.mlm_out_size, device,
                                   **self.dataset_configs[dataset]['tasks'][task])
            self.decoders[task] = decoder

    def forward(self,
                input_token_ids: torch.tensor,
                golds: Dict[str, torch.tensor],
                seg_ids: torch.tensor = None,
                eval_mask: torch.tensor = None,
                offsets: torch.tensor = None,
                subword_mask: torch.tensor = None,
                predicting: bool = False):
        """
        Forward pass
    
        Parameters
        ----------
        input_token_ids: torch.tensor
            Tensor with wordpiece indices. shape=(batch_size, 
            max_sent_len_wordpieces).
        golds: Dict[str, torch.tensor]
            Dictionary with gold labels, keys are the task-names and values
            are the gold labels, dimensions depend on the task-type.
        seg_ids: torch.tensor = None
            Segment id's, also called token_type_ids in the transformers 
            library. Should have the same dimension as input_token_ids:
            (batch_size, max_sent_len_wordpieces).
        eval_mask: torch.tensor = None
            Mask for the tokens/label indices to take into account, 
            shape=(batch_size, max_sent_len_words) filled with 0s and 1s. 
            Not relevant for sentence level tasks. Note that the shape is 
            different from input_token_ids and seg_ids, because we have 
            masks on the word level, not the subword level.
        offsets: torch.tensor = None
            The indices of the wordpieces to use, these can be the first
            or last wordpiece of each token. shape=(batch_size, 
            max_sent_len_words)
        subword_mask: torch.tensor = None
            Mask for the subwords to take into account, 
            shape=(batch_size, max_sent_len_subwords) filled with 0s and 1s. 
        predicting: bool = False
            If predicting, we need to go through all task, otherwise we only
            go through the task present in the gold annotations.

        Returns
        -------
        loss: float
            combined loss over all decoders
        mlm_out_token
            The output embeddings for the tokens, shape=(batch_size, 
            max_sent_len_words, mlm_out_dim). Note that this is on 
            the word level, not the subword level.
        mlm_out_sent
            The output embeddings for the sentences, shape=(batch_size, 
            mlm_out_dim). 
        """
        # detect the task types to handle in this batch
        cur_task_types = []
        for task, task_type in zip(self.tasks, self.task_types):
            if task in golds or task + '-rels' in golds:
                cur_task_types.append(task_type)
        if predicting:
            cur_task_types = self.task_types
        is_only_mlm = sum([task_type != 'mlm' for task_type in cur_task_types]) == 0
        is_only_classification = sum(
            [task_type not in ['classification', 'regression', 'multiclas'] for task_type in cur_task_types]) == 0
        dont_split = is_only_mlm or is_only_classification

        # Run transformer model on input
        mlm_out, mlm_preds = self.encoder.embed(input_token_ids, seg_ids, dont_split, subword_mask)

        mlm_out_sent = None
        mlm_out_token = None
        mlm_out_tok = None

        if 'classification' in self.task_types or 'regression' in self.task_types or 'multiclas' in self.task_types:
            # always take first token, even if it is not a special token
            mlm_out_sent = mlm_out[:, :, :1, :].squeeze(2)
            if self.dropout != None:
                mlm_out_sent = self.dropout(mlm_out_sent)

        if type(offsets) != type(None):
            mlm_out_token = torch.zeros((len(mlm_out), len(offsets), len(offsets[0]), len(mlm_out[0][0][0])),
                                        device=self.device)
            mlm_out_nospecials = mlm_out
            if self.start_token != None:
                mlm_out_nospecials = mlm_out_nospecials[:, :, 1:, :]
            if self.end_token != None:
                mlm_out_nospecials = mlm_out_nospecials[:, :, :-1, :]

            for sentIdx in range(len(offsets)):
                mlm_out_token[:, sentIdx] = mlm_out_nospecials[:, sentIdx, offsets[sentIdx]]
            if self.dropout != None:
                mlm_out_token = self.dropout(mlm_out_token)

        if 'tok' in self.task_types:
            mlm_out_tok = mlm_out
            if self.start_token != None:
                mlm_out_tok = mlm_out_tok[:, :, 1:, :]
            if self.end_token != None:
                mlm_out_tok = mlm_out_tok[:, :, :-1, :]
            if self.dropout != None:
                mlm_out_tok = self.dropout(mlm_out_tok)

        # get loss from all decoders that have annotations
        loss = 0.0
        loss_dict = {}
        if golds != {}:
            for task, task_type in zip(self.tasks, self.task_types):
                if task not in golds and task + '-rels' not in golds: # task not in current dataset
                    continue 
                if task_type in ['classification', 'regression', 'multiclas']:
                    mlm_out_task = myutils.apply_scalar(mlm_out_sent, self.layers[task], self.scalars[task])
                    out_dict = self.decoders[task].forward(mlm_out_task, eval_mask, golds[task])
                elif task_type == 'dependency':
                    mlm_out_task = myutils.apply_scalar(mlm_out_token, self.layers[task], self.scalars[task])
                    out_dict = self.decoders[task].forward(mlm_out_task, eval_mask, golds[task + '-heads'],
                                                               golds[task + '-rels'])
                elif task_type == 'tok':
                    mlm_out_task = myutils.apply_scalar(mlm_out_tok, self.layers[task], self.scalars[task])
                    # We use the subword mask here for evaluation, as every subwords should have
                    # annotation (except the special start/end token).
                    out_dict = self.decoders[task].forward(mlm_out_task, subword_mask[:, self.num_special_tokens:],
                                                               golds[task])
                elif task_type == 'mlm':
                    # Not sure how to apply scalar here, as prediction is already done..
                    out_dict = self.decoders[task].forward(mlm_preds[:, 1:-1, :], golds[task], subword_mask)
                else:
                    mlm_out_task = myutils.apply_scalar(mlm_out_token, self.layers[task], self.scalars[task])
                    out_dict = self.decoders[task].forward(mlm_out_task, eval_mask, golds[task])
                loss += out_dict['loss']
                loss_dict[task] = out_dict['loss'].item()
        return loss, mlm_out_token, mlm_out_sent, mlm_out_tok, mlm_preds, loss_dict

    def get_output_labels(self,
                          input_token_ids: torch.tensor,
                          golds: Dict[str, torch.tensor],
                          seg_ids: torch.tensor = None,
                          eval_mask: torch.tensor = None,
                          offsets: torch.tensor = None,
                          subword_mask: torch.tensor = None, 
                          raw_text: bool=False):
        """
        Run the forward pass, and convert the output indices to labels where
        necessary. 
    
        Parameters
        ----------
        input_token_ids: torch.tensor
            Tensor with wordpiece indices. shape=(batch_size, 
            max_sent_len_wordpieces).
        golds: Dict[str, torch.tensor]
            Dictionary with gold labels, keys are the task-names and values
            are the gold labels, dimensions depend on the task-type.
        seg_ids: torch.tensor = None
            Segment id's, also called token_type_ids in the transformers 
            library. Should have the same dimension as input_token_ids:
            (batch_size, max_sent_len_wordpieces).
        eval_mask: torch.tensor = None
            Mask for the tokens/label indices to take into account, 
            shape=(batch_size, max_sent_len_words) filled with 0s and 1s. 
            Not relevant for sentence level tasks. Note that the shape is 
            different from input_token_ids and seg_ids, because we have 
            masks on the word level, not the subword level.
        offsets: torch.tensor = None
            The indices of the wordpieces to use, these can be the first
            or last wordpiece of each token. shape=(batch_size, 
            max_sent_len_words)
        subword_mask: torch.tensor = None
            Mask for the subwords to take into account, 
            shape=(batch_size, max_sent_len_subwords) filled with 1s and 0s. 
            Only relevant for tokenization task type.
        raw_text:
            No gold annotation available; means here that we predict for all 
            tasks.

        Returns
        -------
        out_dict: Dict[str, List]
            Dictionary with keys=tasks and the values are the list of 
            (lists of) the outputs for this task.
        """
        # Run transformer model on input
        _, mlm_out_token, mlm_out_sent, mlm_out_tok, mlm_preds, _ = self.forward(input_token_ids, {}, seg_ids, 
                                                                                 eval_mask, offsets, 
                                                                                 subword_mask, True)
        out_dict = {}
        has_tok = 'tok' in self.task_types
        if has_tok:
            tok_task = self.tasks[self.task_types.index('tok')]
            mlm_out_tok_merged = myutils.apply_scalar(mlm_out_tok, self.layers[tok_task], self.scalars[tok_task])
            tok_pred = \
                self.decoders[tok_task].get_output_labels(mlm_out_tok_merged, subword_mask[:, self.num_special_tokens:],
                                                          golds[tok_task])['word_labels']

            # This could be done more efficient if a torch tensor was retrieved
            tok_indices = torch.zeros((mlm_out_tok_merged.shape[0], mlm_out_tok_merged.shape[1]), dtype=torch.long,
                                      device=self.device)
            eval_mask = torch.zeros_like(tok_indices)

            for sent_idx in range(len(tok_pred)):
                word_idx = 0
                subword_idx = 0
                for subword_idx in range(len(tok_pred[sent_idx])):
                    if not subword_mask[sent_idx][subword_idx + self.num_special_tokens].item():
                        subword_idx -= 1
                        break
                    if tok_pred[sent_idx][subword_idx] == 'split':
                        tok_indices[sent_idx][word_idx] = subword_idx
                        word_idx += 1
                # Add the last token if it was missed
                if tok_pred[sent_idx][subword_idx] == 'merge':
                    tok_indices[sent_idx][word_idx] = subword_idx
                    word_idx += 1
                eval_mask[sent_idx][:word_idx] = 1

            mlm_out_token = torch.zeros_like(mlm_out_tok)
            for layer_idx in range(len(mlm_out_tok)):
                for sent_idx in range(len(mlm_out_token[0])):
                    length = mlm_out_token.shape[-1]
                    indices = tok_indices[sent_idx][:length]
                    mlm_out_token[layer_idx][sent_idx] = mlm_out_tok[layer_idx][sent_idx][indices]

        for task, task_type in zip(self.tasks, self.task_types):
            if task not in golds and task + '-heads' not in golds: # not a neat location for this?
                if task_type == 'dependency':
                    golds[task + '-heads'] = None
                    golds[task + '-rels'] = None
                else:
                    golds[task] = None
            if not raw_text and task not in golds and task + '-rels' not in golds:
                continue
            if task_type in ['classification', 'regression', 'multiclas']:
                mlm_out_task = myutils.apply_scalar(mlm_out_sent, self.layers[task], self.scalars[task])
                out_dict[task] = self.decoders[task].get_output_labels(mlm_out_task, eval_mask, golds[task])
            elif self.task_types[self.tasks.index(task)] == 'dependency':
                mlm_out_task = myutils.apply_scalar(mlm_out_token, self.layers[task], self.scalars[task])
                if has_tok:
                    out_dict[task] = self.decoders[task].get_output_labels(mlm_out_task, eval_mask)
                else:
                    out_dict[task] = self.decoders[task].get_output_labels(mlm_out_task, eval_mask,
                                                                           golds[task + '-heads'],
                                                                           golds[task + '-rels'])
            elif task_type == 'tok':
                out_dict[task] = {'word_labels': tok_pred}
            elif task_type == 'mlm':
                continue # not sure what to output for MLM task
                #out_dict[task] = self.decoders[task].get_output_labels(mlm_preds, golds[task])
            else:
                mlm_out_task = myutils.apply_scalar(mlm_out_token, self.layers[task], self.scalars[task])
                if has_tok:
                    out_dict[task] = self.decoders[task].get_output_labels(mlm_out_task, eval_mask)
                else:
                    out_dict[task] = self.decoders[task].get_output_labels(mlm_out_task, eval_mask, golds[task])
        return out_dict

    def reset_metrics(self):
        """
        Reset all metrics, in a new epoch, or for a new dataset 
        this should be called (because the metrics are accumalated 
        over batches).
        """
        for decoder in self.decoders:
            self.decoders[decoder].reset_metrics()

    def get_metrics(self):
        """
        Get the metrics for all decoders.

        Returns
        -------
        metrics: Dict[str, float]
            Dictionary with as keys the names of the metric (including
            task name) and as value the score. Includes also a "sum" key, 
            which obviously is the sum over the other metrics.
        """
        metrics = {}
        sum_metrics = 0
        for decoder in self.decoders:
            names, scores, types = self.decoders[decoder].get_metrics()
            for name, score, metric_type in zip(names, scores, types):
                name = decoder + '-' + name
                metrics[name] = score
                # inverse metrics where lower is better
                if metric_type in [Perplexity, AvgDist]:
                    if score == 0.0 and metric_type == Perplexity:
                        continue
                    sum_metrics += 1 / score
                else:
                    sum_metrics += score

        metrics['sum'] = sum_metrics
        return metrics
