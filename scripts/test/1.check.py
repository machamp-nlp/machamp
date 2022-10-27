import os
import json

def getInfo(modelName):
    modelPath = 'logs/' + modelName + '/' 
    if not os.path.isdir(modelPath):
        print('model not found ' + modelName + ' ' + modelPath)
        return {}
    scorePath = modelPath + sorted(os.listdir(modelPath))[-1] + '/metrics.json'
    if not os.path.isfile(scorePath):
        print('scorefile not found ' + scorePath)
        return {}
    return json.load(open(scorePath))

def getScore(modelName, metric, multiply):
    return getInfo(modelName)[metric] * (100 if multiply else 1)

def getTimeRam(modelName):
    info = getInfo(modelName)
    time_tok = info['time_total'].split(':')
    time_total = int(time_tok[1]) + 60 * int(time_tok[0])
    # Assuming that nothing here will take a day
    return time_total, float(info['max_gpu_mem'])

red = '\033[91m'
green = '\033[92m'
black = '\x1b[0m'

def checkScore(modelName, metric, aimScore, opposite=False):
    print(modelName, metric)
    try:
        realScore = getScore(modelName, metric, not opposite)
    except:
        print('score not found')
        return
    color = red
    if realScore > aimScore or (opposite and realScore < aimScore and realScore != 0.0):
        color = green
    if realScore > aimScore:
        print(color + '{:.2f} > {:.2f}'.format(realScore, aimScore) + black)
    else:
        print(color + '{:.2f} < {:.2f}'.format(realScore, aimScore) + black)

def checkTimeRam(name, goal_minutes, goal_gb):
    try:
        real_minutes, real_gb = getTimeRam(name)
    except:
        return
    if real_gb < goal_gb:
        print(green + '{:.2f}'.format(real_gb) + 'gb < ' + str(goal_gb) + 'gb' + black)
    else:
        print(red + '{:.2f}'.format(real_gb) + 'gb > ' + str(goal_gb) + 'gb' + black)

    if real_minutes < goal_minutes:
        print(green + str(real_minutes) + 'mins < ' + str(goal_minutes) + 'mins' + black)
    else:
        print(red + str(real_minutes) + 'mins > ' + str(goal_minutes) + 'mins' + black)
    

# TODO check time/gpu -ram? 
# EWT
checkScore('test.ewt', 'dev_dependency-las', 89)
checkScore('test.ewt', 'dev_feats-acc.', 96)
checkScore('test.ewt', 'dev_upos-acc.', 96)
checkScore('test.ewt', 'dev_lemma-acc.', 97)
checkTimeRam('test.ewt', 45, 5.5)

# NLU
checkScore('test.nlu', 'dev_intent-acc.', 99)
checkScore('test.nlu', 'dev_slots-acc.', 97)
checkTimeRam('test.ewt', 45, 5)

# QNLI
checkScore('test.qnli', 'dev_qnli-acc.', 88)
checkTimeRam('test.qnli', 90, 4.5)

# MLM
checkScore('test.mlm', 'dev_masking-perplexity', 20, True)
checkTimeRam('test.mlm', 20, 8)

# multiseq
checkScore('test.multiseq', 'dev_feats-multi_acc.', 96)
checkTimeRam('test.multiseq', 30, 6)

# crf
checkScore('test.ner', 'dev_ner-span_f1', 82)
checkTimeRam('test.ner', 25, 5)

# regression
checkScore('test.sts', 'dev_sts-b-avg_dist.', 1.5, True)
checkTimeRam('test.sts', 10, 4)


