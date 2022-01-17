from allennlp.common import Params
import random
import os
random.seed(8446)
import myutils

def getLangs(dataset):
    langs = []
    for tsvFile in os.listdir('data/xtreme/download/' + dataset):
        if tsvFile.startswith('dev-'):
            langs.append(tsvFile.split('-')[1].split('.')[0])
    return langs

def pred(name):
    model = myutils.getModel(name)
    if model == '':
        return
    for dataset in ['panx', 'udpos', 'xnli', 'pawsx']:
        for lang in getLangs(dataset):
            outPath = 'preds/' + name + '.' + dataset + '.' + lang
            devSet = 'data/xtreme/download/' + dataset + '/dev-' + lang + '.tsv'
            cmd = 'python3 predict.py ' + model + ' ' + devSet + ' ' + outPath
            cmd += ' --dataset ' + dataset.upper()
            if not os.path.isfile(outPath + '.eval'):
                print(cmd)

cmds = []
for embed in ['mbert', 'rembert']:
    if embed == 'mbert':
        base = Params.from_file('configs/params.json')
        epochs = [15,20]
        batch_sizes = [16,32,64]
    if embed == 'rembert':
        base = Params.from_file('configs/params-rembert.json')
        epochs = [20]
        batch_sizes = [32]
    for epoch in epochs:
        for cut_frac in [.1,.2, .3]:
            for batch_size in batch_sizes:
                for learnRate in [1e-3, 1e-4, 1e-5]:
                    for dropout in [.1,.2,.3]:
                        name = '.'.join([str(x).replace('.','') for x in ['xtreme', embed, cut_frac, learnRate, dropout, batch_size, str(epoch)]])
                        pred(name)

random.shuffle(cmds)
for cmd in cmds:
    print(cmd)


