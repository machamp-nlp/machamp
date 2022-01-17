from allennlp.data.instance import Instance
from allennlp.data import Token
from allennlp.data.fields import TextField, SequenceLabelField, MetadataField, LabelField

import random

def mlm_mask(tokens, start, end, mask_token):
    """
    Does the masking as done in the original BERT: https://www.aclweb.org/anthology/N19-1423.pdf
    Uses -100 for tokens that are not mask, to match the Perplexity evaluation.
    """
    targets = [-100 for _ in tokens]

    for word_idx, token in enumerate(tokens[1:-1]):
        prob = random.random()

        # mask token with 15% probability
        if prob < 0.15:
            prob /= 0.15

            # get gold_label, before it is replaced
            targets[word_idx+1] = token.text_id

            # 80% randomly change token to mask token
            if prob < 0.8:
                tokens[word_idx+1] = mask_token

            # 10% randomly change token to random token 
            # note that we do not even put in the actual word!?
            elif prob < 0.9:
                tokens[word_idx+1] = Token('[MASK]', text_id=random.randint(start+1, end-1), type_id=0)

            # -> rest 10% randomly keep current token

    return targets


def read_unlabeled(dataset, config, path, is_train, max_sents, token_indexers, tokenizer):
    """
    Reads raw data to perform masked language modeling on. This is a separate function, because
    it also already masks the data.
    """
    # TODO make full use of batch size
    data = []
    start_token = tokenizer.tokenize(tokenizer.tokenizer.cls_token)[0]
    end_token = tokenizer.tokenize(tokenizer.tokenizer.sep_token)[0]
    mask_token = tokenizer.tokenize(tokenizer.tokenizer.mask_token)[0]
    skip_first_line = config['skip_first_line']
    
    for sent_idx, sent in enumerate(open(path, encoding='utf-8', mode='r')):
        if skip_first_line:
            skip_first_line = False
            continue
        # skip empty lines
        if len(sent.strip()) == 0:
            continue
        if max_sents != 0 and sent_idx >= max_sents:
            break
    
        # TODO add special tokens automatically?
        tokens = [end_token] + tokenizer.tokenize(sent) + [start_token]
        # TODO R: minimum value of 106 is taken from mbert, hope its sufficient in most cases?
        targets = mlm_mask(tokens, 106, tokenizer.tokenizer.vocab_size, mask_token)
    
        # set them to TOKENIZED, so that they are not tokenized again in the indexer/embedder
        for token in tokens:
            setattr(token, 'ent_type_', 'TOKENIZED')
    
        input_field = TextField(tokens , token_indexers)
        instance = Instance({'tokens': input_field})
    
        if len(config['tasks']) > 1:
            logger.error('currently MaChAmp does not support mlm mixed with other tasks on one dataset')
        task_name = list(config['tasks'].items())[0][0]
    
        instance.add_field(task_name, SequenceLabelField(targets, input_field, label_namespace=task_name))
        
        metadata = {}
        # the other tokens field will often only be available as word-ids, so we save a copy
        metadata['tokens'] = tokens
        metadata["full_data"] = sent
        metadata["col_idxs"] = {task_name:-1}
        metadata['is_train'] = is_train
        metadata['no_dev'] = False
        instance.add_field('metadata', MetadataField(metadata))
    
        data.append(instance)
    
    return data

