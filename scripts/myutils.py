import os

UDversion = '2.8'
seeds = ['1', '2', '3']

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

def hasColumn(path, idx, threshold=.1):
    total = 0
    noWord = 0
    for line in open(path).readlines()[:5000]:
        if line[0] == '#' or len(line) < 2:
            continue
        tok = line.strip().split('\t')
        if tok[idx] == '_':
            noWord += 1
        total += 1
    return noWord/total < threshold

def getModel(name):
    modelDir = 'logs/'
    nameDir = modelDir + name + '/'
    if os.path.isdir(nameDir):
        for modelDir in reversed(os.listdir(nameDir)):
            modelPath = nameDir + modelDir + '/model.tar.gz'
            if os.path.isfile(modelPath):
                return modelPath
    return ''


def getWords(path, max_sents=-1):
    words = []
    if path == '':
        return words
    sents = 0
    for line in open(path):
        if len(line) < 3:
            sents += 1
            continue
        if line[0] == '#':
            continue
        if max_sents != -1 and sents >= max_sents:
            continue
        else:
            words.append(line.split('\t')[1])
    return words

def getOverlap(test, train):
    total = 0
    match = 0
    for word in test:
        total += 1
        if word in train:
            match += 1
    return match/total


