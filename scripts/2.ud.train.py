import os
from allennlp.common import Params
import myutils

if not os.path.isdir('configs/tmp/'):
    os.mkdir('configs/tmp/')

## single-dataset models
for udPath in ['data/ud-treebanks-v' + myutils.UDversion + '.noEUD/', 'data/ud-treebanks-v2.extras.noEUD/']:
    for UDdir in os.listdir(udPath):
        if not UDdir.startswith("UD") or not os.path.isdir(udPath + UDdir):
            continue
        train, dev, test = myutils.getTrainDevTest(udPath + UDdir)
        
        if train != '':
            if not myutils.hasColumn(train, 1, threshold=.1):
                print('noWords ', train)
                continue
            config = {}
            config['train_data_path'] = train
            if dev != '':
                config['validation_data_path'] = dev
            config['word_idx'] = 1
            config['tasks'] = {}
            if myutils.hasColumn(train, 3, threshold=.1):
                config['tasks']['upos'] = {'task_type':'seq', 'column_idx':3}
            if myutils.hasColumn(train, 2, threshold=.95):
                config['tasks']['lemma'] = {'task_type':'string2string', 'column_idx':2}
            if myutils.hasColumn(train, 5, threshold=.95):
                config['tasks']['feats'] = {'task_type':'seq', 'column_idx':5}
            config['tasks']['dependency'] = {'task_type':'dependency', 'column_idx':6}
        
            allennlpConfig = Params({UDdir: config})
            jsonPath = 'configs/tmp/' + UDdir + '.json'
            allennlpConfig.to_file(jsonPath)
            for seed in myutils.seeds:
                print('python3 train.py --dataset_config ' + jsonPath + ' --seed ' + seed + ' --name ' + UDdir + '.' + seed)

def makeConfig(strategy, seed, runNonSmoothed):
    fullConfig = {}
    for udPath in ['data/ud-treebanks-v' + myutils.UDversion + '.noEUD/', 'data/ud-treebanks-v2.extras.noEUD/']:
        for UDdir in os.listdir(udPath):
            if not UDdir.startswith("UD") or not os.path.isdir(udPath + UDdir):
                continue
            train, dev, test = myutils.getTrainDevTest(udPath + UDdir)
            if train != '':
                if not myutils.hasColumn(train, 1, threshold=.1):
                    continue
                config = {}
                config['train_data_path'] = train
                if dev != '':
                    config['validation_data_path'] = dev
                config['word_idx'] = 1
                config['tasks'] = {}
                if strategy in ['concat', 'datasetEmbeds']:
                    if myutils.hasColumn(train, 3, threshold=.1):
                        config['tasks']['upos'] = {'task_type':'seq', 'column_idx':3}
                    if myutils.hasColumn(train, 2, threshold=.95):
                        config['tasks']['lemma'] = {'task_type':'string2string', 'column_idx':2}
                    if myutils.hasColumn(train, 5, threshold=.95):
                        config['tasks']['feats'] = {'task_type':'seq', 'column_idx':5}
                    config['tasks']['dependency'] = {'task_type':'dependency', 'column_idx':6}
                    if strategy == 'datasetEmbeds':
                        config['dataset_embed_idx'] = -1
                elif strategy == 'sepDec':
                    if myutils.hasColumn(train, 3, threshold=.1):
                        config['tasks']['upos-' + UDdir] = {'task_type':'seq', 'column_idx':3}
                    if myutils.hasColumn(train, 2, threshold=.95):
                        config['tasks']['lemma-' + UDdir] = {'task_type':'string2string', 'column_idx':2}
                    if myutils.hasColumn(train, 5, threshold=.95):
                        config['tasks']['feats-' + UDdir] = {'task_type':'seq', 'column_idx':5}
                    config['tasks']['dependency-' + UDdir] = {'task_type':'dependency', 'column_idx':6}
                fullConfig[UDdir] =  config
    
    allennlpConfig = Params(fullConfig)
    jsonPath = 'configs/tmp/fullUD' + strategy + '.json'
    allennlpConfig.to_file(jsonPath)
    cmd = 'python3 train.py --dataset_config ' + jsonPath 
    if strategy == 'datasetEmbeds':
        cmd += ' --parameters_config configs/params.datasetEmbeds.json'
    else:
        cmd += ' --parameters_config configs/params.smoothSampling.json'
    cmd += ' --name fullUD' + strategy + '.smoothed.' + seed + ' --seed ' + seed
    print(cmd)
    if runNonSmoothed:
        cmd = 'python3 train.py --dataset_config ' + jsonPath + ' --name fullUD' + strategy + '.' + seed + ' --seed ' + seed
        print(cmd)


# multi-dataset models
#TODO not efficient, hasColumn is expensive and done 3 times for each check!
for seed in myutils.seeds:
    makeConfig('concat', seed, True)
    makeConfig('sepDec', seed, False)
    makeConfig('datasetEmbeds', seed, False)

    
    
