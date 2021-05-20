posTags = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", '_']

depRels = ["acl", "advcl", "advmod", "amod", "appos", "aux", "case", "cc", "ccomp", "clf", "compound", "conj", "cop", "csubj", "dep", "det", "discourse", "dislocated", "expl", "fixed", "flat", "goeswith", "iobj", "list", "mark", "nmod", "nsubj", "nummod", "obj", "obl", "orphan", "parataxis", "punct", "reparandum", "root", "vocative", "xcomp", '_']

posConvert = {'CONJ':'CCONJ'}
depConvert = {'auxpass': 'aux:pass', 'dobj':'obj', 'nsubjpass':'nsubj:pass', 'name':'flat', 'mwe':'fixed', 'remnant':'conj', 'neg':'advmod', 'csubjpass':'csubj:pass'}
#neg could also be det!

import sys

for line in open(sys.argv[1]):
    tok = line.strip().split('\t')
    if len(tok) < 10:
        print(line.strip())
    else:
        upos = tok[3]
        deprel_short = tok[7].split(':')[0]
        deprel = tok[7]
        if len(deprel_short) > 1:
            deprel = deprel.replace('_', ':')
        if deprel in depConvert:
            deprel = depConvert[deprel]
        if upos in posConvert:
            upos = posConvert[upos]
        tok[3] = upos
        tok[7] = deprel
        print('\t'.join(tok))


