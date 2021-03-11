import os

posTags = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN", "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X", '_']

depRels = ["acl", "advcl", "advmod", "amod", "appos", "aux", "case", "cc", "ccomp", "clf", "compound", "conj", "cop", "csubj", "dep", "det", "discourse", "dislocated", "expl", "fixed", "flat", "goeswith", "iobj", "list", "mark", "nmod", "nsubj", "nummod", "obj", "obl", "orphan", "parataxis", "punct", "reparandum", "root", "vocative", "xcomp", '_']

def getTrainDevTest(path):
    dev = ''
    test = ''
    train = ''
    path = path + '/'
    for conlFile in os.listdir(path):
        if 'conll' in conlFile:
            if 'train' in conlFile:
                train = path + conlFile
            if 'dev' in conlFile:
                dev = path + conlFile
            if 'test' in conlFile:
                test = path + conlFile
    return train, dev, test


for udDir in os.listdir('.'):
    if os.path.isdir(udDir):
        train, dev, test = getTrainDevTest(udDir)

        for conlPath in [train,dev,test]:
            if conlPath == '':
                continue
            # official check
            print(conlPath)
            cmd = 'python3 conll18_ud_eval.py ' + conlPath + ' ' + conlPath 
            os.system(cmd)

            # check UPOS/relations
            notFound = {}
            for line in open(conlPath):
                tok = line.strip().split('\t')
                if len(tok) < 10:
                    continue
                upos = tok[3]
                deprel = tok[7].split(':')[0]
                if upos not in posTags and upos:
                    if upos not in notFound:
                        notFound[upos] = 1
                    else:
                        notFound[upos] += 1
                if deprel not in depRels and deprel:
                    if deprel not in notFound:
                        notFound[deprel] = 1
                    else:
                        notFound[deprel] += 1
            for item in notFound:
                print(item, notFound[item])
