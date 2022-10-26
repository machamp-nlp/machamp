import inspect
import logging
import math

import torch

logger = logging.getLogger(__name__)

from transformers import AutoModel


class MachampEncoder:
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
        self.num_extra_tokens = 2 - [start_token_id, end_token_id].count(None)

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
            decoder_start_token_id = self.mlm.config.bos_token_id or self.mlm.config.decoder_start_token_id
            decoder_input_ids = torch.full((batch_size, 1), decoder_start_token_id, device=input_token_ids.device)
            args['decoder_input_ids'] = decoder_input_ids
            # an alternative is to use input_token_ids, then one can use decoder_hidden states below
            # on EWT one would get slightly higher scores, but also use 30% more gpu ram

        output = self.mlm.forward(**args)

        # Copy over language modeling predictions if they are present.
        # Note that this will usually (always?) be the case when mlm
        # is an AutoModelForMaskedLM. 
        logits = None
        if hasattr(output, 'logits'):
            logits = output.logits

        all_layers = None
        if hasattr(output, 'hidden_states'):
            all_layers = output.hidden_states
        # elif hasattr(output, 'decoder_hidden_states'):
        #    all_layers = output.encoder_hidden_states
        elif hasattr(output, 'encoder_hidden_states'):
            all_layers = output.encoder_hidden_states
        else:
            logger.error(
                'Error, not sure how to extract hidden states from the encoder of ' + self.mlm.name_or_path +
                ' of type ' + str(type(self.mlm)))
            exit(1)

        # In case of different sizes, we only keep layers with the same size as the last layer
        # for efficiency. This happens for example in google/canine-c
        layers_to_consider = []
        for layer in all_layers:
            if layer.shape == all_layers[-1].shape:
                layers_to_consider.append(layer)
        # Shape= num_layers:batch_size:max_tokens:emb_size
        return torch.stack(layers_to_consider), logits

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

                # Split
                batch_size = input_token_ids.size(0)
                find_end_token = 0 if self.end_token_id == None else self.end_token_id
                # get lengths, note that they include special tokens
                lengths = []
                for sent_idx in range(batch_size):
                    if find_end_token in input_token_ids[sent_idx]:
                        lengths.append((torch.nonzero(input_token_ids[sent_idx] == find_end_token)[0]).item() + 1)
                    else:
                        lengths.append(len(input_token_ids[sent_idx]))

                # We make the batches longer, but this has only a small
                # effect on memory usage, as a maximum number of words per
                # batch is used
                amount_of_splits = [self.get_size(length, self.max_input_length) for length in lengths]
                new_batch_size = sum(amount_of_splits)

                new_input_tokens = torch.full((new_batch_size, self.max_input_length), find_end_token,
                                              device=input_token_ids.device, dtype=torch.int64)
                new_seg_ids = torch.full((new_batch_size, self.max_input_length), 0, device=input_token_ids.device,
                                         dtype=torch.int64)
                # at version 0.4, this would never happen, all sequence level
                # tasks have a subword mask, and all others don't get in here
                # (they would truncate instead of merge). If we need this `if`
                # we would also need many more below though
                # if type(subword_mask) != type(None):
                new_subword_mask = torch.full((new_batch_size, self.max_input_length), 0,
                                              device=input_token_ids.device, dtype=torch.int64)
                if self.start_token_id != None:
                    new_input_tokens[:, 0] = self.start_token_id

                cur_batch_idx = 0
                num_subwords_per_batch = self.max_input_length - self.num_extra_tokens
                for sent_idx in range(batch_size):
                    # if current sentence < max_len, just copy it
                    if lengths[sent_idx] <= self.max_input_length:
                        new_input_tokens[cur_batch_idx][:lengths[sent_idx]] = input_token_ids[sent_idx][
                                                                              :lengths[sent_idx]]
                        new_seg_ids[cur_batch_idx][:lengths[sent_idx]] = seg_ids[sent_idx][:lengths[sent_idx]]
                        new_subword_mask[cur_batch_idx][:lengths[sent_idx]] = subword_mask[sent_idx][:lengths[sent_idx]]
                        cur_batch_idx += 1
                    else:
                        # remove special tokens for simplicity, we will add them in each split manually
                        beg_idx = 0 if self.start_token_id == None else 1
                        end_idx = None if self.end_token_id == None else -1
                        token_ids_sent = input_token_ids[sent_idx][beg_idx:end_idx]
                        seg_ids_sent = seg_ids[sent_idx][beg_idx:end_idx]
                        subword_mask_sent = subword_mask[sent_idx][beg_idx:end_idx]
                        for split in range(amount_of_splits[sent_idx]):
                            src_beg = num_subwords_per_batch * split
                            tgt_beg = beg_idx
                            if split + 1 == amount_of_splits[sent_idx]:
                                src_end = lengths[sent_idx] - self.num_extra_tokens
                                tgt_end = (lengths[sent_idx] - self.num_extra_tokens) - (
                                        split * num_subwords_per_batch) + tgt_beg
                            else:
                                src_end = num_subwords_per_batch * (split + 1)
                                tgt_end = self.max_input_length - (1 if self.end_token_id != None else 0)

                            new_input_tokens[cur_batch_idx][tgt_beg:tgt_end] = token_ids_sent[src_beg:src_end]
                            new_seg_ids[cur_batch_idx][tgt_beg:tgt_end] = seg_ids_sent[src_beg:src_end]
                            new_subword_mask[cur_batch_idx][tgt_beg:tgt_end] = subword_mask_sent[src_beg:src_end]
                            if self.start_token_id != None:
                                # Copy these from first subword to special start token
                                # Note that the start token in new_input_tokens was already set earlier
                                new_seg_ids[cur_batch_idx][0] = new_seg_ids[cur_batch_idx][1]
                                new_subword_mask[cur_batch_idx][0] = new_subword_mask[cur_batch_idx][1]
                            if self.end_token_id != None:
                                # Copy these from last subword to special end token
                                # Note that the end token in new_input_tokens was already set earlier
                                new_seg_ids[cur_batch_idx][tgt_end] = new_seg_ids[cur_batch_idx][tgt_end - 1]
                                new_subword_mask[cur_batch_idx][tgt_end] = new_subword_mask[cur_batch_idx][tgt_end - 1]

                            cur_batch_idx += 1

                # Embed:
                mlm_out_split, mlm_preds = self.run_mlm(new_input_tokens, new_seg_ids, new_subword_mask)

                # Merge
                num_layers = len(mlm_out_split)
                mlm_out_merged = torch.zeros(num_layers, batch_size, input_token_ids.size(1), mlm_out_split.size(-1),
                                             device=input_token_ids.device, dtype=torch.float32)

                splitted_idx = 0  # rename this is the global (/batch) index, split_idx is per sentence/utterance)
                for sent_idx in range(batch_size):
                    if amount_of_splits[sent_idx] == 1:
                        mlm_out_merged[:, sent_idx, 0:lengths[sent_idx]] = mlm_out_split[:, splitted_idx,
                                                                           0:lengths[sent_idx]]
                        splitted_idx += 1
                    else:
                        for split_idx in range(amount_of_splits[sent_idx]):
                            src_beg = 0 if (split_idx == 0 or self.start_token_id == None) else 1
                            src_end = self.max_input_length - (1 if self.end_token_id != None else 0)
                            tgt_beg = split_idx * (self.max_input_length - self.num_extra_tokens) + (
                                1 if self.start_token_id != None and split_idx != 0 else 0)
                            tgt_end = tgt_beg + num_subwords_per_batch + (
                                1 if self.start_token_id != None and split_idx == 0 else 0)
                            if split_idx == amount_of_splits[sent_idx] - 1:
                                # amount of total subwords % amount of subwords per split
                                tgt_end = lengths[sent_idx]
                                src_end = lengths[sent_idx] - tgt_beg
                                if self.end_token_id != None:
                                    src_end += 1
                            mlm_out_merged[:, sent_idx, tgt_beg:tgt_end] = mlm_out_split[:, splitted_idx,
                                                                           src_beg:src_end]
                            splitted_idx += 1

                # Note that mlm_preds is not split. This is an error/bug, but we hardcoded that for the MLM
                # task splitting shouldn't happen in the reader (its size is always < max_length), so it will 
                # never occur in practice
                return mlm_out_merged, mlm_preds
