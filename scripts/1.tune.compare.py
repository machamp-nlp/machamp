#from allennlp.common import Params
import os
import myutils
import json
import ast

def getLangs(dataset):
    langs = []
    for tsvFile in os.listdir('data/xtreme/download/' + dataset):
        if tsvFile.startswith('dev-'):
            langs.append(tsvFile.split('-')[1].split('.')[0])
    return langs

datasets = ['panx', 'udpos', 'xnli', 'pawsx']

def getIDscore(name):
    scores = []
    for dataset in datasets:
        scoreFile = 'preds/' + name + '.' + dataset + '.en.eval'
        score = ast.literal_eval(' '.join(open(scoreFile).readlines()))['.run/.sum']
        scores.append(score)
    return sum(scores)/len(scores)

def getOODscore(name):
    scores = []
    for dataset in datasets:
        for language in getLangs(dataset):
            scoreFile = 'preds/' + name + '.' + dataset + '.' + language + '.eval'
            score = ast.literal_eval(' '.join(open(scoreFile).readlines()))['.run/.sum']
            scores.append(score)
    return sum(scores)/len(scores)

IDscores = {}
OODscores = {}
cmds = []
for embed in ['mbert', 'rembert']:
    if embed == 'mbert':
        #epochs = [15,20]
        epochs = [20]
        batch_sizes = [16,32,64]
    if embed == 'rembert':
        epochs = [20]
        batch_sizes = [32]
    for epoch in epochs:
        for cut_frac in [.1,.2, .3]:
            for batch_size in batch_sizes:
                for learnRate in [1e-3, 1e-4, 1e-5]:
                    for dropout in [.1,.2,.3]:
                        name = '.'.join([str(x).replace('.','') for x in ['xtreme', embed, cut_frac, learnRate, dropout, batch_size, epoch]])
                        IDscores[name] = getIDscore(name)
                        OODscores[name] = getOODscore(name)


def getScores(paramVal, nameIdx, scores):
    total = []
    for embed in ['mbert', 'rembert']:
        embScores = []
        for setting in scores:
            tok = setting.split('.')
            if tok[nameIdx] == str(paramVal).replace('.','') and tok[1] == embed:
                embScores.append(scores[setting])
        if embScores == []:
            total.append(0.0)
        else:
            total.append(100*sum(embScores)/len(embScores))
    return total

print(' & '.join(['param', 'val', 'IDmbert', 'OODmbert', 'IDrembert', 'OODrembert']) + '\\\\')
for setup in [('batch_size', [16,32,64], 5), ('cut_frac', [.1, .2, .3], 2), ('learnRate', [1e-3, 1e-4, 1e-5], 3), ('dropout', [.1,.2,.3], 4)]: #('epochs',[15,20], 6)
    name = setup[0]
    paramRange = setup[1]
    nameIdx = setup[2]
    for param in paramRange:
        IDmbert, IDrembert = getScores(param, nameIdx, IDscores)
        OODmbert, OODrembert = getScores(param, nameIdx, OODscores)
        print(' & '.join([name, str(param)] + ['{:.2f}'.format(x) for x in [IDmbert, OODmbert, IDrembert, OODrembert]]) + '\\\\')



