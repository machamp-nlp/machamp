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
                 end_token_id: int,
                 start_token_id: int):
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
        end_token_id: int
            The token id used for padding (behind the input)
        start_token_id: int
            The token id used for the start-of-sentence token (also
            called the cls token since BERT)
        """
        self.mlm = mlm
        self.max_input_length = max_input_length
        self.end_token_id = end_token_id
        self.start_token_id = start_token_id
        self.num_extra_tokens = 2-[start_token_id, end_token_id].count(None)

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
        return max(1, math.ceil((own_size - self.num_extra_tokens) / (max_size - self.num_extra_tokens)))

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
        args = {'input_ids': input_token_ids, 'attention_mask': subword_mask, 'output_hidden_states': True}
        argspec = inspect.getfullargspec(self.mlm.forward)
        # detect whether token_type_ids (segment embeddings) are supported
        if 'token_type_ids' in argspec[0]:
            args['token_type_ids'] = seg_ids
        if 'decoder_input_ids' in argspec[0]:
            batch_size = len(input_token_ids)
            decoder_start_token_id = self.mlm.config.bos_token_id
            if decoder_start_token_id == None:
                decoder_start_token_id = self.mlm.config.decoder_start_token_id
            decoder_input_ids = torch.ones((batch_size, 1), dtype=torch.long, device=input_token_ids.device) * decoder_start_token_id
            args['decoder_input_ids'] = decoder_input_ids

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

        all_layers = None
        if hasattr(output, 'hidden_states'):
            all_layers = output.hidden_states
        elif hasattr(output, 'decoder_hidden_states'):
            all_layers = output.decoder_hidden_states
        elif hasattr(output, 'encoder_hidden_states'):
            all_layers = output.encoder_hidden_states
        else:
            logger.error(
                'Error, not sure how to extract hidden states from the encoder of ' + self.mlm.name_or_path + ' of type ' + str(
                    type(self.mlm)))
            exit(1)
        # Shape= num_layers:batch_size:max_tokens:emb_size
        return torch.stack(all_layers), logits

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
        depending on the batch. We do not use a sliding window at the moment
        for readabilities sake (still failed to make the code readable 
        unforunately ;( ).

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
                # Shall we add the special last token and lose one subword instead?
                return self.run_mlm(input_token_ids[:, :self.max_input_length], seg_ids[:, :self.max_input_length],
                                    subword_mask[:, :self.max_input_length])
            else:  # split, embed, merge
                batch_size = input_token_ids.size(0)
                if self.end_token_id != None:
                    lengths = [(torch.nonzero(input_token_ids[sent_idx] == self.end_token_id)[0]).item() + 1 for
                           sent_idx in range(batch_size)]
                else:
                    lengths = []
                    for sent_idx in range(batch_size):
                        if 0 in input_token_ids[sent_idx]:
                            lengths.append((torch.nonzero(input_token_ids[sent_idx] == 0)[0]).item() + 1)
                        else:
                            lengths.append(len(input_token_ids[sent_idx]))

                amount_of_splits = [self.get_size(length, self.max_input_length) for length in lengths]
                new_batch_size = sum(amount_of_splits)
                if self.end_token_id != None:
                    new_input_tokens = torch.full((new_batch_size, self.max_input_length), self.end_token_id,
                                                  device=input_token_ids.device, dtype=torch.int64)
                else:
                    new_input_tokens = torch.full((new_batch_size, self.max_input_length), 0,
                                                  device=input_token_ids.device, dtype=torch.int64)
                new_seg_ids = torch.full((new_batch_size, self.max_input_length), 0, device=input_token_ids.device, dtype=torch.int64)
                if type(subword_mask) != type(None):
                    new_subword_mask = torch.full((new_batch_size, self.max_input_length), 0, device=input_token_ids.device, dtype=torch.int64)
                curBatchIdx = 0
                for sentIdx in range(batch_size):
                    # if current sentence < max_len, just copy it
                    if lengths[sentIdx] <= self.max_input_length:
                        new_input_tokens[curBatchIdx][:lengths[sentIdx]] = input_token_ids[sentIdx][:lengths[sentIdx]]
                        new_seg_ids[curBatchIdx][:lengths[sentIdx]] = seg_ids[sentIdx][:lengths[sentIdx]]
                        new_subword_mask[curBatchIdx][:lengths[sentIdx]] = subword_mask[sentIdx][:lengths[sentIdx]]
                        curBatchIdx += 1
                    else:
                        # remove special tokens for simplicity, we will add them in each split manually
                        token_ids_sent = input_token_ids[sentIdx]
                        seg_ids_sent = seg_ids[sentIdx]
                        if type(subword_mask) != type(None):
                            subword_mask_sent = subword_mask[sentIdx]

                        if self.start_token_id != None:
                            token_ids_sent = token_ids_sent[1:]
                            seg_ids_sent = seg_ids_sent[1:]
                            if type(subword_mask) != type(None):
                                subword_mask_sent = subword_mask_sent[1:]
                        if self.end_token_id != None:
                            token_ids_sent = token_ids_sent[:-1]
                            seg_ids_sent = seg_ids_sent[:-1]
                            if type(subword_mask) != type(None):
                                subword_mask_sent = subword_mask_sent[:-1]

                        for split in range(amount_of_splits[sentIdx]):
                            beg = (self.max_input_length - self.num_extra_tokens) * split
                            if split + 1 == amount_of_splits[sentIdx]:
                                end = lengths[sentIdx]-self.num_extra_tokens
                            else:
                                end = (self.max_input_length - self.num_extra_tokens) * (split + 1)
                            if self.start_token_id != None:
                                new_input_tokens[curBatchIdx][1:end - beg + 1] = token_ids_sent[beg:end]
                                new_input_tokens[curBatchIdx][0] = self.start_token_id
                                new_seg_ids[curBatchIdx][1:end - beg + 1] = seg_ids_sent[beg:end]
                                new_seg_ids[curBatchIdx][0] = new_seg_ids[curBatchIdx][1]
                                new_subword_mask[curBatchIdx][0] = 1
                                new_subword_mask[curBatchIdx][1:end - beg + 1] = subword_mask_sent[beg:end]
                                new_subword_mask[curBatchIdx][0] = 1
                                new_subword_mask[curBatchIdx][1:end - beg + 1] = subword_mask_sent[beg:end]
                            else:
                                new_input_tokens[curBatchIdx][:end - beg] = token_ids_sent[beg:end]
                                new_seg_ids[curBatchIdx][:end - beg] = seg_ids_sent[beg:end]
                                new_subword_mask[curBatchIdx][:end - beg] = subword_mask_sent[beg:end]
                                new_subword_mask[curBatchIdx][:end - beg] = subword_mask_sent[beg:end]

                            curBatchIdx += 1

                # We make the batches longer, but this has only a small
                # effect on memory usage, as a maximum number of words per
                # batch is used
                mlm_out_split, mlm_preds = self.run_mlm(new_input_tokens, new_seg_ids, new_subword_mask)
                num_layers = len(mlm_out_split)
                if self.end_token_id != None:
                    mlm_out_merged = torch.full((num_layers, batch_size, input_token_ids.size(1), mlm_out_split.size(-1)), self.end_token_id,
                                             device=input_token_ids.device, dtype=torch.float32)
                else:
                    mlm_out_merged = torch.zeros(num_layers, batch_size, input_token_ids.size(1), mlm_out_split.size(-1),
                                             device=input_token_ids.device, dtype=torch.float32)
                for layer_idx in range(num_layers):
                    splitted_idx = 0
                    for sent_idx in range(batch_size):
                        if amount_of_splits[sent_idx] == 1:
                            mlm_out_merged[layer_idx][sent_idx][0:lengths[sent_idx]] = mlm_out_split[layer_idx][splitted_idx][0:lengths[sent_idx]]
                            splitted_idx += 1
                        else:
                            # first of the splits, keep as is
                            end_idx = self.max_input_length-1 if self.end_token_id == None else self.max_input_length
                            # It would be neater to merge this into the line above
                            if self.num_extra_tokens == 0:
                                end_idx += 1
                            mlm_out_merged[layer_idx][sent_idx][0:end_idx] = mlm_out_split[layer_idx][splitted_idx][0:end_idx]
                            num_subwords_per_batch = self.max_input_length - self.num_extra_tokens

                            splitted_idx += 1
                            # all except first and last, has no CLS/SEP
                            for i in range(1, amount_of_splits[sent_idx] - 1):
                                beg = end_idx + (i-1) * num_subwords_per_batch
                                end = beg + num_subwords_per_batch
                                mlm_out_cursplit = mlm_out_split[layer_idx][splitted_idx]
                                if self.end_token_id != None:
                                    mlm_out_cursplit = mlm_out_cursplit[:-1]
                                if self.start_token_id != None:
                                    mlm_out_cursplit = mlm_out_cursplit[1:]
        
                                mlm_out_merged[layer_idx][sent_idx][beg:end] = mlm_out_cursplit
                                splitted_idx += 1

                            # last of the splits, keep the SEP
                            beg = end_idx + (amount_of_splits[sent_idx]-2) * num_subwords_per_batch
                            end = lengths[sent_idx]
                            mlm_out_merged[layer_idx][sent_idx][beg:end] = mlm_out_split[layer_idx][splitted_idx][0:end - beg]
                            splitted_idx += 1
                # Note that mlm_preds is not split. This is an error/bug, but we hardcoded that for the MLM
                # task splitting shouldn't happen in the reader (its size is always < max_length), so it will 
                # never occur in practice
                return mlm_out_merged, mlm_preds
