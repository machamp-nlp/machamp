import os

def getTrainDevTest(path):
    train = ''
    dev = ''
    test = ''
    for conlFile in os.listdir(path):
        if conlFile.endswith('conllu'):
            if 'train' in conlFile:
                train = path + '/' + conlFile
            if 'dev' in conlFile:
                dev = path + '/' + conlFile
            if 'test' in conlFile:
                test = path + '/' + conlFile
    return train, dev, test

def hasColumn(path, idx):
    total = 0
    noWord = 0
    for line in open(path).readlines()[:5000]:
        if line[0] == '#' or len(line) < 2:
            continue
        tok = line.strip().split('\t')
        if tok[idx] == '_':
            noWord += 1
        total += 1
    return noWord/total < .1

    return True

def getModel(name):
    modelDir = 'logs/'
    nameDir = modelDir + name + '/'
    if os.path.isdir(nameDir):
        for modelDir in reversed(os.listdir(nameDir)):
            modelPath = nameDir + modelDir + '/model.tar.gz'
            if os.path.isfile(modelPath):
                return modelPath
    return ''

