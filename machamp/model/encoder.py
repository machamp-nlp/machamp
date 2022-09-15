import inspect
import logging
import math

import torch

logger = logging.getLogger(__name__)

from transformers import AutoModel


class MachampEncoder():
    def __init__(self,
                 mlm: AutoModel,
                 max_input_length: int,
                 padding_token_id: int,
                 cls_token_id: int):
        """
        The main (shared) encoder of a MachampModel. This class
        is mainly handling the formatting of the input/output to
        the AutoModel.

        Parameters
        ----------
        mlm: transformers.AutoModel
            The language model used as encoder, it is already
            initialized in the MachampModel, as over there we 
            need quite some information from it.
        max_input_length: int
            The maximum input length to the encoder, most of the
            code in this class is actually to handle this correctly.
        padding_token_id: int
            The token id used for padding (behind the input)
        cls_token_id: int
            The token id used for the start-of-sentence token (also
            called the cls token since BERT)
        """
        self.mlm = mlm
        self.max_input_length = max_input_length
        self.padding_token_id = padding_token_id
        self.cls_token_id = cls_token_id

    def get_size(self, own_size: int, max_size: int):
        """
        This converts the size of an instance (sentence) to the 
        number of splits it should be split in to comply with 
        the maximum size (maxSize). We take into account that
        each instance should start and end with a special token, 
        and assume that these are included in the sizes (hence 
        the -2's). 

        Parameters
        ----------
        own_size: int
            The size of the instance in question
        max_size: int
            The maximum size it is allowed to have

        Returns
        -------
        num_splits: int
            the amount of splits necessary to make ownSize comply with maxSize
        """
        # max(1, ..) is necessary for empty inputs, we do not want
        # to have 0 splits!
        return max(1, math.ceil((own_size - 2) / (max_size - 2)))

    def run_mlm(self,
                input_token_ids: torch.tensor,
                seg_ids: torch.tensor,
                subword_mask: torch.tensor):
        """ 
        Runs self.mlm (an AutoModel), and return the last state
        of the encoder. Note that input should already be 
        truncated to self.max_input_length here

        Parameters
        ----------
        input_token_ids: torch.tensor
            Tensor with wordpiece indices. shape=(batch_size, 
            max_input_lenght).
        seg_ids: torch.tensor
            Segment id's, also called token_type_ids in the transformers 
            library. Should have the same dimension as input_token_ids:
            (batch_size, max_input_length).
        subword_mask: torch.tensor = None
            Mask for the subwords to take into account, 
            shape=(batch_size, max_input_length) filled with 0s and 1s. 

        Returns
        -------
        embedded_text: torch.tensor
            shape=(batch_size,max_sent_len,emb_size)
        """
        # first detect whether token_type_ids (segment embeddings) are supported
        argspec = inspect.getfullargspec(self.mlm.forward)
        args = {'input_ids': input_token_ids, 'attention_mask': subword_mask, 'output_hidden_states': True}
        if 'token_type_ids' in argspec[0]:
            args['token_type_ids'] = seg_ids

        output = self.mlm.forward(**args)

        # Copy over language modeling predictions if they are present.
        # Note that this will usually (always?) be the case when mlm
        # is an AutoModelForMaskedLM. We will however not always need
        # it, in those cases. For example when we do MLM on one dataset
        # and another task on another, it is copied always for now.
        # Could potentially save space/time by detecting whether we need
        # it?
        logits = None
        if hasattr(output, 'logits'):
            logits = output.logits

        if hasattr(output, 'hidden_states'):
            return output.hidden_states[-1], logits
        elif hasattr(output, 'last_hidden_state'):
            return output.last_hidden_state, logits
        elif hasattr(output, 'decoder_last_hidden_state'):
            return output.decoder_last_hidden_state, logits
        elif hasattr(output, 'encoder_last_hidden_state'):
            return output.encoder_last_hidden_state, logits
        else:
            logger.error(
                'Error, not sure how to extract last hidden state from the encoder of ' + self.mlm.name_or_path + ' of type ' + str(
                    type(self.mlm)))
            exit(1)

    def embed(self,
              input_token_ids: torch.tensor,
              seg_ids: torch.tensor,
              dont_split: bool,
              subword_mask: torch.tensor = None):
        """
        Embeds the token ID's from input_token_ids. This splits the input
        sentences that are longer than self.max_input_length, and merges
        their outputs afterwards. We do it this way because it costs a lot
        of memory in the transformers library, for the decoders this matters
        a lot less, so we can already merge here. For the descriptions of 
        the parameter below, note that max_sent_len_wordpieces is a variable, 
        depending on the batch.

        Parameters 
        ----------
        input_token_ids: torch.tensor
            Tensor with wordpiece indices. shape=(batch_size, 
            max_sent_len_wordpieces).
        seg_ids: torch.tensor
            Segment id's, also called token_type_ids in the transformers 
            library. Should have the same dimension as input_token_ids:
            (batch_size, max_sent_len_wordpieces).
        dont_split: bool
            Normally we would split by max_input_length, but for some
            tasks (i.e. sentence level tasks), this doesnt make much
            sense, and we just remove any tokens beyond max_input_length.
        subword_mask: torch.tensor = None
            Mask for the subwords to take into account, 
            shape=(batch_size, max_sent_len_subwords) filled with 0s and 1s. 

        Returns
        -------
        embedded_text: torch.tensor
            shape=(batch_size,max_sent_len,emb_size)
        """
        # input is smaller than max_len, so just embed and return
        if input_token_ids.size(-1) <= self.max_input_length:
            return self.run_mlm(input_token_ids, seg_ids, subword_mask)
        else:  # input is too long, handle:
            if dont_split:  # truncate
                return self.run_mlm(input_token_ids[:, :self.max_input_length], seg_ids[:, :self.max_input_length],
                                    subword_mask[:, :self.max_input_length])
            else:  # split, embed, merge
                batch_size = input_token_ids.size(0)

                lengths = [(torch.nonzero(input_token_ids[sent_idx] == self.padding_token_id)[0]).item() + 1 for
                           sent_idx in range(batch_size)]
                amount_of_splits = [self.get_size(length, self.max_input_length) for length in lengths]
                new_batch_size = sum(amount_of_splits)
                new_input_tokens = torch.full((new_batch_size, self.max_input_length), self.padding_token_id,
                                              device=input_token_ids.device, dtype=torch.int64)
                new_seg_ids = torch.full((new_batch_size, self.max_input_length), 0, device=input_token_ids.device, dtype=torch.int64)
                new_subword_mask = torch.full((new_batch_size, self.max_input_length), 0, device=input_token_ids.device, dtype=torch.int64)
                curBatchIdx = 0
                for sentIdx in range(batch_size):
                    if lengths[sentIdx] <= self.max_input_length:
                        new_input_tokens[curBatchIdx][:lengths[sentIdx]] = input_token_ids[sentIdx][:lengths[sentIdx]]
                        new_seg_ids[curBatchIdx][:lengths[sentIdx]] = seg_ids[sentIdx][:lengths[sentIdx]]
                        new_subword_mask[curBatchIdx][:lengths[sentIdx]] = subword_mask[sentIdx][:lengths[sentIdx]]
                        curBatchIdx += 1
                    else:
                        # remove special tokens for simplicity, then we can just take max_input_length-2 elements
                        # for each split (except the last)
                        token_ids_sent = input_token_ids[sentIdx][1:-1]
                        seg_ids_sent = seg_ids[sentIdx][1:-1]
                        if type(subword_mask) != type(None):
                            subword_mask_sent = subword_mask[sentIdx][1:-1]
                        for split in range(amount_of_splits[sentIdx]):
                            beg = (self.max_input_length - 2) * split
                            if split + 1 == amount_of_splits[sentIdx]:
                                end = lengths[sentIdx]-2
                            else:
                                end = (self.max_input_length - 2) * (split + 1)
                            new_input_tokens[curBatchIdx][1:end - beg + 1] = token_ids_sent[beg:end]
                            new_input_tokens[curBatchIdx][0] = self.cls_token_id
                            new_seg_ids[curBatchIdx][1:end - beg + 1] = seg_ids_sent[beg:end]
                            new_seg_ids[curBatchIdx][0] = new_seg_ids[curBatchIdx][1]
                            new_subword_mask[curBatchIdx][0] = 1
                            new_subword_mask[curBatchIdx][1:end - beg + 1] = subword_mask_sent[beg:end]
                            curBatchIdx += 1

                # would it make sense to split it first?, instead of 35*max_len, have 32*max_len and 3*max_len 
                # and then run the mlm twice?
                # AllenNLP doesn't to do this, and its much easier without, so for now we leave it
                mlm_out_split, mlm_preds = self.run_mlm(new_input_tokens, new_seg_ids, new_subword_mask)
                mlm_out_merged = torch.zeros(batch_size, input_token_ids.size(1), mlm_out_split.size(-1),
                                             device=input_token_ids.device)
                splitted_idx = 0
                for sent_idx in range(batch_size):
                    if amount_of_splits[sent_idx] == 1:
                        mlm_out_merged[sent_idx][0:lengths[sent_idx]] = mlm_out_split[splitted_idx][0:lengths[sent_idx]]
                        splitted_idx += 1
                    else:
                        # first of the splits, keep the CLS
                        mlm_out_merged[sent_idx][0:self.max_input_length - 1] = mlm_out_split[splitted_idx][
                                                                                0:self.max_input_length - 1]
                        splitted_idx += 1
                        # all except first and last, only keep the body (not CLS, not SEP)
                        for i in range(1, amount_of_splits[sent_idx] - 1):
                            beg = i * (
                                        self.max_input_length - 2) - 1  # -1 because the first line doesnt have a SEP, -2 because we do not need CLS and SEP from each split
                            end = beg + self.max_input_length - 2
                            mlm_out_merged[sent_idx][beg:end] = mlm_out_split[splitted_idx][1:-1]
                            splitted_idx += 1

                        # last of the splits, keep the SEP
                        beg = (amount_of_splits[sent_idx] - 1) * (self.max_input_length - 2) - 1
                        end = lengths[sent_idx]-1
                        mlm_out_merged[sent_idx][beg:end] = mlm_out_split[splitted_idx][0:end - beg]
                        splitted_idx += 1
                # Note that mlm_preds is not split. This is an error/bug, but we hardcoded that for the MLM
                # task, splitting shouldn't happen, so it will never occur in practice
                return mlm_out_merged, mlm_preds
