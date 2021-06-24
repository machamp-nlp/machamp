import os
import myutils 
from allennlp.common import Params

tgtDir = 'data/gigantamax/'

if not os.path.isdir(tgtDir):
    os.mkdir(tgtDir)


trainConfig = {}
for udPath in ['data/ud-treebanks-v' + myutils.UDversion + '.noEUD/', 'data/ud-treebanks-v2.extras.noEUD/']:
    for UDdir in os.listdir(udPath):
        if not (os.path.isdir(udPath + UDdir) and UDdir.startswith('UD')):
            continue
        train, dev, test = myutils.getTrainDevTest(udPath + UDdir)
        if not myutils.hasColumn(test, 1, threshold=.1):
            continue
        config = {}

        newUDdir = tgtDir + UDdir
        trainPath = tgtDir + UDdir + '.traintest'
        devPath = tgtDir + UDdir + '.dev'
        cmd = 'cat ' + train + ' ' + test + ' > ' + trainPath
        os.system(cmd)
        if dev != '':
            cmd = 'cat ' + dev + ' > ' + devPath
            os.system(cmd)
            config['validation_data_path'] = devPath

        config['train_data_path'] = trainPath
        config['word_idx'] = 1
        config['tasks']  = {}
        if myutils.hasColumn(trainPath, 3, threshold=.1):
            config['tasks']['upos'] = {'task_type':'seq', 'column_idx':3}
        if myutils.hasColumn(trainPath, 2, threshold=.95):
            config['tasks']['lemma'] = {'task_type':'string2string', 'column_idx':2}
        if myutils.hasColumn(trainPath, 5, threshold=.95):
            config['tasks']['feats'] = {'task_type':'seq', 'column_idx':5}
        config['tasks']['dependency'] = {'task_type':'dependency', 'column_idx':6}
        
        trainConfig[UDdir] = config
        
        
allennlpConfig = Params(trainConfig)
jsonPath = 'configs/tmp/gigantamax.json'
allennlpConfig.to_file(jsonPath)
cmd = 'python3 train.py --dataset_config ' + jsonPath
cmd += ' --parameters_config configs/params.smoothSampling.json'
print(cmd)

