import random

from allennlp.data.tokenizers import Token


def seqs2data(conllu_file, do_lowercase):
    """
    Reads a conllu-like file. We do not base the comment identification on
    the starting character being a '#' , as in some of the datasets we used
    the words where in column 0, and could start with a `#'. Instead we start
    at the back, and see how many columns (tabs) the file has. Then we judge
    any sentences at the start which do not have this amount of columns (tabs)
    as comments. Returns both the read column data as well as the full data.
    """
    sent = []
    for line in open(conllu_file, mode="r", encoding="utf-8"):
        if do_lowercase:
            line = line.lower()
        # because people use paste command, which includes empty tabs
        if len(line) < 2 or line.replace('\t', '') == '':
            if len(sent) == 0:
                continue
            num_cols = len(sent[-1])
            beg_idx = 0
            for i in range(len(sent)):
                back_idx = len(sent) - 1 - i
                if len(sent[back_idx]) == num_cols:
                    beg_idx = len(sent) - 1 - i
            yield sent[beg_idx:], sent
            sent = []
        else:
            sent.append([token for token in line[:-1].split('\t')])

    # adds the last sentence when there is no empty line
    if len(sent) != 0 and sent != ['']:
        num_cols = len(sent[-1])
        beg_idx = 0
        for i in range(len(sent)):
            back_idx = len(sent) - 1 - i
            if len(sent[back_idx]) == num_cols:
                beg_idx = len(sent) - 1 - i
        yield sent[beg_idx:], sent


def lines2data(input_file, do_lowercase):
    """
    Simply reads a tab-separated text file. Returns each line split
    by a '\t' character.
    """
    for line in open(input_file, mode='r', encoding='utf-8'):
        if len(line.strip()) < 2:
            continue
        if do_lowercase:
            line = line.lower()
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
