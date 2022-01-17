import random
import unicodedata

from allennlp.data.tokenizers import Token
from transformers.tokenization_utils import _is_whitespace
from transformers.tokenization_utils import _is_control

def lines2data(input_file, skip_first_line=False):
    """
    Simply reads a tab-separated text file. Returns each line split
    by a '\t' character.
    """
    for line in open(input_file, mode='r', encoding='utf-8'):
        if skip_first_line:
            skip_first_line = False
            continue
        if len(line.strip()) < 2:
            continue
        tok = [part for part in line.strip().split('\t')]
        yield tok


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


def _clean_text(text):
    """Performs invalid character removal and whitespace cleanup on text."""
    output = []
    for char in text:
        cp = ord(char)
        if cp == 0 or cp == 0xFFFD or _is_control(char):
            continue
        if _is_whitespace(char):
            output.append(" ")
        else:
            output.append(char)
    return "".join(output)


