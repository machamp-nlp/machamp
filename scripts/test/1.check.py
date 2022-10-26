import os
import json
from termcolor import colored

def getScore(modelName, metric):
    modelPath = 'logs/' + modelName + '/' 
    if not os.path.isdir(modelPath):
        return 0.0
    scorePath = modelPath + sorted(os.listdir(modelPath))[-1] + '/metrics.json'
    print(scorePath)
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
checkScore('test.ewt', 'best_dev_dependency-las', 89)
checkScore('test.ewt', 'best_dev_feats-acc', 97)
checkScore('test.ewt', 'best_dev_upos-acc', 96)
checkScore('test.ewt', 'best_dev_lemma-acc', 98)

# NLU
checkScore('test.nlu', 'best_dev_intent-acc', 99)
checkScore('test.nlu', 'best_dev_slots-acc', 97)

# QNLI
checkScore('test.qnli', 'best_dev_snli-acc', 89)

# MLM
checkScore('test.mlm', 'best_dev_mlm-ppl', 10, True)

# multiseq
checkScore('test.multiseq', 'ner/multi_span_f1', 60)

# crf
checkScore('test.crf', 'best_dev_ner-span_f1', 70)

# all

# raw_input

# long instances?

    
