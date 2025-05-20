import logging
import os
from typing import List

import torch
from transformers import AutoTokenizer
from transformers import tokenization_utils
from transformers.models.bert.tokenization_bert import BasicTokenizer

from machamp.utils.lemma_edit import min_edit_script

logger = logging.getLogger(__name__)

class ScriptFinder:
    def __init__(self):
        """
        This class can be used to identify the script of a text, it uses the unicode standard.
        It automatically downloads the indices of the scripts in the unicode ranges. 
        """
        ranges = []
        if not os.path.isfile('scripts/Scripts.txt'):
            os.system('mkdir -p scripts')
            os.system('wget https://www.unicode.org/Public/15.0.0/ucd/Scripts.txt --no-check-certificate -O scripts/Scripts.txt')
        for line in open('scripts/Scripts.txt'):
            tok = line.split(';')
            if line[0]!='#' and len(tok) == 2:
                char_range_hex = tok[0].strip().split('..')
                char_range_int = [int(x, 16) for x in char_range_hex]
                if len(char_range_int) == 1:
                    char_range_int.append(char_range_int[0])
                # Note that we include the first and the last character of the range 
                # in the indices, so the first range for Latin is 65-90 for example, 
                # character 65 (A) and 90 (Z) are both included in the Latin set.
                # This means that for single characters (caught in the "if" above)
                # the same number is repeated twice
                ranges.append(char_range_int + [tok[1].strip().split()[0]])

        self.ranges = sorted(ranges)

    def find_char(self, char: chr):
        """
        Finds the script of a single character

        Parameters
        ----------
        char: chr
            The input character for which we want to find the script

        Returns
        -------
        script: str
            The script as defined by the unicode standard. None if not found.
        """
        if len(char) > 1:
            char = char[0]
        char_idx = ord(char)
        for rangeIdx, char_range in enumerate(self.ranges):
            if char_idx >= char_range[0]:
                if char_idx <= char_range[1]:
                    return char_range[2]
            if char_range[1] > char_idx: # we can give up, because the list is sorted
                break
        return None

    def guess_script(self, text: str):
        """
        Finds the script of a text, by finding the script of each character and
        returning the most commonly used script that is not ``Common''.
        
        Parameters
        ----------
        text: str
            input text

        Returns
        -------
        script: str
            The (majority) script of the text
        """
        classes = {}
        for char in text:
            cat = self.find_char(char)
            if cat == None or cat == 'Common':
                continue
            if cat not in classes:
                classes[cat] = 0
            classes[cat] += 1
        main_class = sorted(classes.items(), key=lambda x: x[1], reverse=True)[0][0]
        return main_class


def _find_additional_splits(text: str, script_finder: ScriptFinder, do_splits: bool):
    """
    Finds indices of japanese characters (Hiragana/Katakama), Kanji is already included 
    in the basictokenizer. Also finds indices of script changes, which solves for example
    splitting 20s -> 20 s, or 5$ -> 5 $

    Parameters
    ----------
    text: str
        The input text
    script_finder: ScriptFinder
        Can be used to find the script of a character
    do_splits: bool
        Whether to split on script switches

    Returns
    -------
    indices_jap: List[int]
        indices of japanese characters
    indices_script: List[int]
        indices of where scripts change
    """
    indices_jap = []
    indices_script = []
    prev_script = None
    for charIdx, char in enumerate(text):
        script = script_finder.find_char(char)
        if script in ['Hiragana', 'Katakama']:
            indices_jap.append(charIdx)
            #indices_jap.append(charIdx+1)# Performs is negligibly lower with this, but the models becomes slower
        elif do_splits and prev_script != None and prev_script != script:
            indices_script.append(charIdx)
        prev_script = script
    return indices_jap,indices_script

def insert(form_list, tgt_char_idx, char_to_add):
    """
    Removes a character at a certain character index in a list of subwords.

    Parameters
    ----------
    form_list: List
        List of subwords
    tgt_char_idx: int
        index of character to remove (note that index comes from full text, not a list)
    char_to_add: chr
        The character that we want to insert
    Returns
    -------
    form_list: List
        List of subwords, with 1 character less
    """
    counter = 0
    for subwordIdx, subword in enumerate(form_list):
        if counter + len(subword) >= tgt_char_idx > counter:
            word_char_idx = tgt_char_idx - counter
            # If this is the last character, and the following subword is empty, put it
            # there instead
            # the following subword is probably empty, because it just has been removed?
            # This could be checked if one wants to be extra careful (by looking at the edit rule)
            if subwordIdx < len(form_list)-1 and len(form_list[subwordIdx+1]) == 0  and word_char_idx == len(form_list[subwordIdx]):
                form_list[subwordIdx+1] = char_to_add
            # If this is the first character, and the previous subword is empty, put it
            # there instead
            # the previous subword is probably empty, because it just has been removed?
            # This could be checked if one wants to be extra careful (by looking at the edit rule)
            elif subwordIdx > 0 and len(form_list[subwordIdx-1]) == 0 and word_char_idx == 0:
                form_list[subwordIdx-1] = char_to_add
            else:
                form_list[subwordIdx] = form_list[subwordIdx][:word_char_idx] + char_to_add + form_list[subwordIdx][word_char_idx:]

        counter += len(subword)

    if tgt_char_idx == 0:# this was not caught in the for loop above
        if len(form_list) == 0:
            form_list = [char_to_add]
        else:
            form_list[0] = char_to_add + form_list[0]

    return form_list

def remove(form_list: List[str], tgt_char_idx: int):
    """
    Removes a character at a certain character index in a list of subwords.

    Parameters
    ----------
    form_list: List
        List of subwords
    tgt_char_idx: int
        index of character to remove (note that index comes from full text, not a list)
    Returns
    -------
    form_list: List
        List of subwords, with 1 character less
    """
    counter = 0
    for subwordIdx, subword in enumerate(form_list):
        if counter + len(subword) >= tgt_char_idx > counter:
            word_char_idx = tgt_char_idx-counter
            form_list[subwordIdx] = form_list[subwordIdx][:word_char_idx-1] + form_list[subwordIdx][word_char_idx:]
        counter += len(subword)
    return form_list
    
def apply_edit_rule(rule: str, form_list: List[str]):
    """
    Applies the edit rule to the form to generate the original character sequence.
    Note that is slightly complicated by the fact that the text is in a list of 
    subwords.
    
    Parameters
    ----------
    rule: str
        The rule as generated by the min_edit_script
    form_list: List
        List of (sub)words

    Returns
    -------
    form_list: List
        
    """
    tgt_char_idx = 0
    j = 0
    while j < len(rule):
        if rule[j] == "→":
            tgt_char_idx += 1
        elif rule[j] == "-":
            form_list = remove(form_list, tgt_char_idx+1)
        else:
            assert (rule[j] == "+")
            form_list = insert(form_list, tgt_char_idx, rule[j+1])
            tgt_char_idx += 1
            j += 1
        j += 1
    return form_list

def get_space_locations(text: str):
    """
    Find the indices of whitespaces. Note that the whitespaces
    themselves are not counted in the indices.

    Parameters
    ----------
    text: str
        Input text with whitespaces

    Returns
    -------
    space_locations: List[int]
        Location of whitespaces in text
    """
    space_locs = []
    cur_char_idx = 0
    for char in text:
        if char == ' ':
            space_locs.append(cur_char_idx)
        else:
            cur_char_idx += 1
    space_locs.append(cur_char_idx)
    return space_locs

def clean_whitespace(text: str):
    """
    Replace all non standard whitespace with a standard whitespace, 
    also remove duplicate whitespaces.

    Parameters
    ----------
    text: str
        Input text (possibly with non-standard whitespaces)
    
    Returns
    -------
    text: str
        Text with normalized whitespaces
    """
    cleaned_text = []
    for char in text:
        if tokenization_utils._is_whitespace(char):
            cleaned_text.append(" ")
        else:
            cleaned_text.append(char)
    return "".join(cleaned_text).replace('  ', ' ')


def tok(orig: str, pre_tokenizer: BasicTokenizer, tokenizer: AutoTokenizer, pre_splits: dict, script_finder: ScriptFinder, do_splits: bool, type_tokenizer: str):
    """
    
    Parameters
    ----------
    orig: str
        The original input (completely untokenized)
    pre_tokenizer: BasicTokenizer
        The tokenizer that should split the punctuation
    tokenizer: AutoTokenizer
        The subword segmenter
    pre_splits: dict
        Words that has to be splits, including their gold splits (with whitespaces)
    script_finder: ScriptFinder
        Can be used to identify in which script a character/text is written (here 
        used for chars only)
    do_splits:
        Whether to do additional splits on locations where scripts switch.
    type_tokenizer: str
        One of ['subword', 'sentencepiece', 'other'], we only support the first 2
        We need this information to know how handle ## and ▁

    Returns
    -------
    no_unk_subwords: List[str]
        Subwords, de-normalized, de-unked and completely aligned to the chars in orig
    token_ids: List[int]
        Token ids obtained from orig
    pre_tokked_split: List[str]
        Punctuation split only tokenization
    """
    # This is necessary cleaning, as whitespace are considered equal in UD, but not in most LMs
    orig = clean_whitespace(orig)
    # pre_tokenization (separate punctuation tokens
    pre_tokked = pre_tokenizer.tokenize(orig)

    # create token_ids and no_unk_subwords
    no_unk_subwords = []
    token_ids = []
    for word in pre_tokked:
        jap_indices, script_diff_indices = _find_additional_splits(word, script_finder, do_splits)
        if word in pre_splits:
            split = pre_splits[word]
            indices = [index for index in range(len(split)) if split.startswith(' ', index)]
            script_diff_indices.extend(indices)
        all_indices = jap_indices + script_diff_indices

        if all_indices != []:
            tokked = []
            for index in reversed(all_indices):
                word = word[:index] + ' ' +word[index:]
            for subpart in word.split(' '):
                tokked_subpart = tokenizer.tokenize(subpart)
                if tokked_subpart == [tokenizer.unk_token]:
                    tokked.append(subpart)
                    token_ids.append(tokenizer.unk_token_id)
                elif tokenizer.unk_token in tokked_subpart:
                    logger.error("somehow a (part of an) unknown token has a length of more than a subword, this might disrupt the output of the tokenization.")
                    pass
                else:
                    for subword in tokked_subpart:
                        # Second part is because performance for Japanese went down a lot when using the ##
                        # which is odd.
                        prefix_to_id = tokenizer.convert_tokens_to_ids(['##' + subword]) 
                        if type_tokenizer == 'wordpiece' and len(script_diff_indices) > len(jap_indices) and len(prefix_to_id) == 1 and prefix_to_id != [tokenizer.unk_token_id]:
                            tokked.append('##' + subword)
                            token_ids.append(prefix_to_id[0])
                        else:
                            tokked.append(subword)
                            token_ids.append(tokenizer.convert_tokens_to_ids(subword))
        else:
            tokked = tokenizer.tokenize(word)
            token_ids.extend(tokenizer.convert_tokens_to_ids(tokked))

        if tokenizer.unk_token in tokked:
            no_unk_subwords.append(word)
            if len(tokked) > 1:            
                logger.error("somehow a (part of an) unknown token has a length of more than 3 subwords, this might disrupt the output of the model.")
                logger.error(word)
                exit(1)
        else:
            for subword in tokked:
                if type_tokenizer == 'sentencepiece':
                    no_unk_subwords.append(subword.replace('▁', ''))
                elif type_tokenizer == 'wordpiece':
                    no_unk_subwords.append(subword[2:] if subword.startswith('##') else subword)
                elif type_tokenizer == 'G':
                    no_unk_subwords.append(subword.replace('Ġ', ''))
                else:
                    logger.error('error, type of tokenizer unknown. The tokenization task currently only supports tokenizers that use ##, Ġ,  or ▁')
                    exit(1)

    # Clean (de-normalize) the no_unk_subwords, 
    no_unk_subwords_chars = ''.join(no_unk_subwords)
    orig_chars = orig.replace(' ', '')
    if no_unk_subwords_chars !=  orig_chars:
        edit_rule = min_edit_script(no_unk_subwords_chars, orig_chars)
        no_unk_subwords = apply_edit_rule(edit_rule, no_unk_subwords)
        if ''.join(no_unk_subwords) !=  orig_chars:
            logger.error('Error, characters do not match for:')
            logger.error(''.join(no_unk_subwords))
            logger.error(orig.replace(' ', ''))
    if len(no_unk_subwords) != len(token_ids):
        no_unk_subwords = [x for x in no_unk_subwords if x != '']
        if len(no_unk_subwords) != len(token_ids):
            logger.error('Tokenization annotation has wrong length:')
            logger.error(no_unk_subwords)
            logger.error(token_ids)
    return no_unk_subwords, token_ids, pre_tokked
        
def get_splits(subwords: List[str], space_idxs: List[int]):
    """
    Generate splits based on list of predicted subwords and character
    indices where gold tokens have a split
    
    Parameters
    ----------
    subwords: List
        List of automatically tokenized subwords
    space_idxs: List
        Characted indices where splits were made in the gold

    Returns
    -------
    splits: List
        list of tuples that contain the original word and the split version (with whitespaces)
    """
    splits = []
    cur_char_idx = 0
    for subword in subwords:
        new_subword = subword
        rel_pos_splits = [spaceIdx - cur_char_idx for spaceIdx in space_idxs]
        for split_idx in reversed(rel_pos_splits):
            if len(new_subword) > split_idx > 0:
                new_subword = new_subword[:split_idx] + ' ' + new_subword[split_idx:]
        cur_char_idx += len(subword)

        #if new_subword != subword:
        splits.append([subword, new_subword])
    return splits

def get_offsets(gold_spaces: List[int], tokked_spaces: List[int]):
    """
    Find the offsets based on the splits (whitespaces) from gold and tokked.
    This might not always be perfectly possible, but we try to find the best
    here. (This could also mean that a token is used multiple times)

    Parameters
    ----------
    gold_spaces: List
        The location of whitespaces (/splits) in the gold tokens
    tokked_spaces: List
        The location of whitespaces (/splits) in the predicted tokenization

    Returns
    -------
    offsets: torch.tensor
        The indices of each gold token in the predicted tokenization
    """
    offsets = []
    for spaceLocation in gold_spaces:
        if spaceLocation in tokked_spaces:
            # get first match, the second one refers to the whitespace (▁)
            offsets.append(tokked_spaces.index(spaceLocation))
        else:
            # Try to match with word before and after
            # match with the closest one of both, if it is not already used
            # else, match with the furthest one. If both are already used, 
            # match with the closest one
            guess_loc = 0
            for guess_loc in range(len(tokked_spaces)):
                if tokked_spaces[guess_loc] > spaceLocation:
                    break
            if tokked_spaces[-1] < spaceLocation:
                options = tokked_spaces[-1:]
            else:
                options = tokked_spaces[guess_loc-1 if guess_loc > 0 else 0:guess_loc+1]
            out_gold = [option for option in options if option not in gold_spaces]
            if len(out_gold) == 1: # only one not used in gold, use it
                offsets.append(tokked_spaces.index(out_gold[0]))
            else: # they are both in or out of gold, find closest
                dists = [abs(spaceLocation-estimate) for estimate in options]
                pos_in_options = dists.index(min(dists))
                offsets.append(tokked_spaces.index(options[pos_in_options]))
    return torch.tensor(offsets, dtype=torch.long)

def to_gold(offsets: List[int], length: int):
    """
    We rely on the fact that the offsets for word level 
    tasks align with the places where we want to tokenize.
    So converting offsets to tokenization labels is trivial
    
    Parameters
    ----------
    offsets: List
        Offsets of the gold tokens in the predicted tokenization
    length: int
        The total length of the token_ids, so that we know how long the
        annotation should be.

    Returns
    -------
    annotation: List[str]
        Annotation based on offsets
    """
    annotation = ['merge'] * length
    for split_idx in offsets:
        annotation[split_idx] = 'split'
    return annotation

def tokenize_and_annotate(
        full_data: List[List[str]], 
        gold: List[str], 
        pre_tokenizer: BasicTokenizer, 
        tokenizer: AutoTokenizer, 
        pre_splits: dict, 
        learn_new_splits: bool, 
        script_finder: ScriptFinder, 
        do_splits: bool, 
        type_tokenizer: str):
    """
    Tokenizes and generates annotation for tokenization. The complication
    comes from the fact that each LM has different normalization. We use
    the min_edit script to generate a list of matching subwords, not containing
    any normalization, unknown tokens and/or whitespaces (no_unk_subwords).
    Then, we find all whitespaces (/splits) from the gold tokenization and the
    automatic tokenization, which we use to get the offsets/annotation.

    Parameters
    ----------
    full_data: List
        List of tokens and their annotations
    gold: List
        Gold tokenization
    pre_tokenizer: BasicTokenizer
        Tokenizer to split puncutation
    tokenizer: AutoTokenizer
        Subword tokenizer, aligned with LM to use
    pre_splits: dict
        Splits to do before the subword segmentation
    learn_new_splits: bool
        Whether we have to learn new splits
    script_finder: ScriptFinder
        Can be used to find the script of a character
    do_splits: bool
        Whether to split on script switches
    type_tokenizer: str
        One of ['subword', 'sentencepiece', 'other'], we only support the
        first two for now.

    Returns
    -------
    token_ids: List
        token_ids from gold, to use as input for LM
    offsets: torch.tensor
        (Estimated) indices of gold tokens
    tok_gold: List
        Generated annotation for tokenization
    no_unk_subwords: List
        Output of tokenization, but then de-normalized and de-unked (and no whitespace)
    pre_splits: dict
        Containing splits not found by the pre_tokenizer.
    """
    orig = ''
    for line in full_data:
        if line[0].startswith('# text =') and len(line[0]) > 9:
            orig = line[0][8:].strip()
        if line[0].startswith('# text  =') and len(line[0]) > 9:
            orig = line[0][9:].strip()
        if line[0].startswith('# text=') and len(line[0]) > 9:
            orig = line[0][8:].strip()
    if orig == '':
        logger.error(
            'No original text found in file, altough tokenization task type is used. Make sure you have a comment in '
            'front of each instance that starts with "# text = ".')
        for line in full_data:
            logger.error('\t'.join(line))
        exit(1)

    no_unk_subwords, token_ids, pre_tokked = tok(orig, pre_tokenizer, tokenizer, pre_splits, script_finder, do_splits, type_tokenizer)
    if ''.join(no_unk_subwords) != ''.join(gold).replace(' ', ''):
        logger.error("Error; somehow the original input does not match the gold characters:")
        logger.error(''.join(no_unk_subwords) + ' != \n' + ''.join(gold).replace(' ', ''))

    gold = [word.replace(' ', '') for word in gold]
    gold_spaces = get_space_locations(' '.join(gold))
    tokked_spaces = get_space_locations(' '.join(no_unk_subwords))
    offsets = get_offsets(gold_spaces, tokked_spaces)
    tok_gold = to_gold(offsets, len(no_unk_subwords))
    
    new_splits = {}
    if learn_new_splits:
        not_found_splits = [spaceIdx for spaceIdx in gold_spaces if spaceIdx not in tokked_spaces]
        found_splits = get_splits(pre_tokked, not_found_splits)
        # In the future we can save counts and use a heuristic to select?
        for split in found_splits: 
            if split[0] != split[1]: 
                new_splits[split[0]] = split[1]

    pre_splits.update(new_splits)
    # Apply own pre-split and re-tokenize:
    # Note that splits learned from other sentences (further down), could also affect
    # the tokenization here. We ignore this in favor of efficiency and simplicity
    # (otherwise we would have to tokenize twice and/or search for matches later), 
    # the effect should be small anyways.
    if learn_new_splits and len(new_splits) > 0:
        return tokenize_and_annotate(full_data, gold, pre_tokenizer, tokenizer, pre_splits, False, script_finder, do_splits, type_tokenizer)

    return token_ids, offsets, tok_gold, no_unk_subwords, pre_splits

