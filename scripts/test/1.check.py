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
    return getInfo(modelName)['best_' + metric] * (100 if multiply else 1)

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
    print()    

# EWT
checkScore('test.ewt', 'dev_dependency_las', 89)
checkScore('test.ewt', 'dev_feats_accuracy', 96)
checkScore('test.ewt', 'dev_upos_accuracy', 96)
checkScore('test.ewt', 'dev_lemma_accuracy', 97)
checkTimeRam('test.ewt', 45, 5.5)

# NLU
checkScore('test.nlu', 'dev_intent_accuracy', 99)
checkScore('test.nlu', 'dev_slots_accuracy', 97)
checkTimeRam('test.ewt', 45, 5)

# QNLI
checkScore('test.qnli', 'dev_qnli_accuracy', 88)
checkTimeRam('test.qnli', 100, 4.5)

# MLM
checkScore('test.mlm', 'dev_masking_perplexity', 2, True)
checkTimeRam('test.mlm', 60, 8)

# multiseq
checkScore('test.multiseq', 'dev_feats_multi_acc', 96)
checkTimeRam('test.multiseq', 30, 6)

# crf
checkScore('test.ner', 'dev_ner_span_f1', 82)
checkTimeRam('test.ner', 25, 5)

# regression
checkScore('test.sts', 'dev_sts-b_avg_dist', 1.5, True)
checkTimeRam('test.sts', 10, 4)

# multitask
checkScore('test.multitask', 'dev_dependency_las', 89)
checkScore('test.multitask', 'dev_feats_accuracy', 96)
checkScore('test.multitask', 'dev_upos_accuracy', 96)
checkScore('test.multitask', 'dev_lemma_accuracy', 97)
checkScore('test.multitask', 'dev_qnli_accuracy', 88)
checkTimeRam('test.multitask', 145, 5.5)

checkScore('test.multitask-sequential.0', 'dev_dependency_las', 89)
checkScore('test.multitask-sequential.0', 'dev_feats_accuracy', 96)
checkScore('test.multitask-sequential.0', 'dev_upos_accuracy', 96)
checkScore('test.multitask-sequential.0', 'dev_lemma_accuracy', 97)
checkTimeRam('test.multitask-sequential.0', 45, 5.5)
checkScore('test.multitask-sequential.1', 'dev_qnli_accuracy', 88)
checkTimeRam('test.multitask-sequential.1', 100, 5.5)

checkScore('test.multitask-div0', 'dev_dependency_las', 88)
checkScore('test.multitask-div0', 'dev_feats_accuracy', 96)
checkScore('test.multitask-div0', 'dev_upos_accuracy', 96)
checkScore('test.multitask-div0', 'dev_lemma_accuracy', 97)
checkScore('test.multitask-div0', 'dev_qnli_accuracy', 89)
checkTimeRam('test.multitask-div0', 500, 9)

checkScore('test.multitask-div1', 'dev_dependency_las', 88)
checkScore('test.multitask-div1', 'dev_feats_accuracy', 96)
checkScore('test.multitask-div1', 'dev_upos_accuracy', 96)
checkScore('test.multitask-div1', 'dev_lemma_accuracy', 97)
checkScore('test.multitask-div1', 'dev_qnli_accuracy', 90)
checkTimeRam('test.multitask-div1', 275, 7)

checkScore('test.multitask-nodiv0', 'dev_dependency_las', 87)
checkScore('test.multitask-nodiv0', 'dev_feats_accuracy', 96)
checkScore('test.multitask-nodiv0', 'dev_upos_accuracy', 96)
checkScore('test.multitask-nodiv0', 'dev_lemma_accuracy', 97)
checkScore('test.multitask-nodiv0', 'dev_qnli_accuracy', 89)
checkTimeRam('test.multitask-nodiv0', 275, 5.5)

checkScore('test.multitask-nodiv1', 'dev_dependency_las', 86)
checkScore('test.multitask-nodiv1', 'dev_feats_accuracy', 96)
checkScore('test.multitask-nodiv1', 'dev_upos_accuracy', 95)
checkScore('test.multitask-nodiv1', 'dev_lemma_accuracy', 95)
checkScore('test.multitask-nodiv1', 'dev_qnli_accuracy', 89)
checkTimeRam('test.multitask-nodiv1', 175, 5.5)

