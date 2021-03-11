import os
import json
from termcolor import colored

def getScore(modelName, metric):
    modelPath = 'logs/' + modelName + '/' 
    if not os.path.isdir(modelPath):
        return 0.0
    scorePath = modelPath + sorted(os.listdir(modelPath))[-1] + '/metrics.json'
    if not os.path.isfile(scorePath):
        return 0.0
    return json.load(open(scorePath))['best_validation_.run/' + metric] * 100

def checkScore(modelName, metric, aimScore, opposite=False):
    print(modelName, metric)
    realScore = getScore(modelName, metric)
    color = 'red'
    if realScore > aimScore or (opposite and realScore < aimScore and realScore != 0.0):
        color = 'green'
    if realScore > aimScore:
        print(colored('{:.2f} > {:.2f}'.format(realScore, aimScore), color))
    else:
        print(colored('{:.2f} < {:.2f}'.format(realScore, aimScore), color))
    # TODO check if output is actually written!        

# EWT
checkScore('test.ewt', 'dependency/las', 89)
checkScore('test.ewt', 'feats/acc', 97)
checkScore('test.ewt', 'upos/acc', 96)
checkScore('test.ewt', 'xpos/acc', 96)
checkScore('test.ewt', 'lemma/acc', 98)

# NLU
checkScore('test.nlu', 'intent/acc', 99)
checkScore('test.nlu', 'slots/acc', 97)

# QNLI
checkScore('test.qnli', 'snli/acc', 89)

# MLM
checkScore('test.mlm', 'mlm/ppl', 10, True)

# NMT
checkScore('test.nmt', 'en-nl/bleu', 20)

# multiseq
checkScore('test.multiseq', 'ner/multi_span_f1', 60)

# crf
checkScore('test.crf', 'ner/span_f1', 70)

# all

# raw_input

# long instances?

    
