import os
from allennlp.common import Params
import myutils


def makeConfig(strategy):
    udPath = 'data/ud-treebanks-v2.7/'
    fullConfig = {}
    for UDdir in os.listdir(udPath):
        train, dev, test = myutils.getTrainDevTest(udPath + UDdir)
        if train != '':
            if not myutils.hasColumn(train, 1):
                continue
            config = {}
            config['train_data_path'] = train
            if dev != '':
                config['validation_data_path'] = dev
            config['word_idx'] = 1
            config['tasks'] = {}
            if strategy in ['concat', 'datasetEmbeds']:
                config['tasks']['upos'] = {'task_type':'seq', 'column_idx':3}
                if myutils.hasColumn(train, 2)  and UDdir not in ['UD_Arabic-PADT', 'UD_Korean-Kaist', 'UD_Estonian-EDT', 'UD_Breton-KEB']:
                    config['tasks']['lemma'] = {'task_type':'string2string', 'column_idx':2}
                config['tasks']['feats'] = {'task_type':'seq', 'column_idx':5}
                config['tasks']['dependency'] = {'task_type':'dependency', 'column_idx':6}
                if strategy == 'datasetEmbeds':
                    config['dataset_embed_idx'] = -1
            elif strategy == 'sepDec':
                config['tasks']['upos-' + UDdir] = {'task_type':'seq', 'column_idx':3}
                if myutils.hasColumn(train, 2)  and UDdir not in ['UD_Arabic-PADT', 'UD_Korean-Kaist', 'UD_Estonian-EDT', 'UD_Breton-KEB']:
                    config['tasks']['lemma-' + UDdir] = {'task_type':'string2string', 'column_idx':2}
                config['tasks']['feats-' + UDdir] = {'task_type':'seq', 'column_idx':5}
                config['tasks']['dependency-' + UDdir] = {'task_type':'dependency', 'column_idx':6}
            fullConfig[UDdir] =  config
    
    allennlpConfig = Params(fullConfig)
    jsonPath = 'configs/fullUD' + strategy + '.json'
    allennlpConfig.to_file(jsonPath)
    cmd = 'python3 train.py --dataset_config ' + jsonPath 
    cmd += ' --parameters_config configs/params.smoothSampling.json'
    cmd += ' --name fullUD' + strategy + '.smoothed'
    print(cmd)
    cmd = 'python3 train.py --dataset_config ' + jsonPath 
    print(cmd)


#TODO not efficient, hasColumn is expensive and done 3 times for each check!
makeConfig('concat')
makeConfig('sepDec')
makeConfig('datasetEmbeds')



# single training
if not os.path.isdir('configs/tmp/'):
    os.mkdir('configs/tmp/')

udPath = 'data/ud-treebanks-v2.7/'
for UDdir in os.listdir(udPath):
    train, dev, test = myutils.getTrainDevTest(udPath + UDdir)
    
    if train != '':
        if not myutils.hasColumn(train, 1):
            continue
        config = {}
        config['train_data_path'] = train
        if dev != '':
            config['validation_data_path'] = dev
        config['word_idx'] = 1
        config['tasks'] = {}
        config['tasks']['upos'] = {'task_type':'seq', 'column_idx':3}
        if myutils.hasColumn(train, 2)  and UDdir not in ['UD_Arabic-PADT', 'UD_Korean-Kaist', 'UD_Estonian-EDT', 'UD_Breton-KEB']:
            config['tasks']['lemma'] = {'task_type':'string2string', 'column_idx':2}
        config['tasks']['feats'] = {'task_type':'seq', 'column_idx':5}
        config['tasks']['dependency'] = {'task_type':'dependency', 'column_idx':6}
    
        allennlpConfig = Params({UDdir: config})
        jsonPath = 'configs/tmp/' + UDdir + '.json'
        allennlpConfig.to_file(jsonPath)
        print('python3 train.py --device 1 --dataset_config ' + jsonPath)


