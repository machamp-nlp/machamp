import os
import myutils

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

udPath = 'data/ud-treebanks-v2.7/'
UDWORDS = {}
for UDdir in os.listdir(udPath):
    train, dev, test = myutils.getTrainDevTest(udPath + UDdir)
    if not myutils.hasColumn(test, 1):
        continue
    words = getWords(train)
    UDWORDS[UDdir] = set(words)

def getOverlap(test, train):
    total = 0
    match = 0
    for word in test:
        total += 1
        if word in train:
            match += 1
    return match/total

PROXIES = {}
for UDdir in os.listdir(udPath): 
    train, dev, test = myutils.getTrainDevTest(udPath + UDdir)
    if dev == '':
        testWords = getWords(test, 10)
        scores = {}
        for proxy in UDWORDS:
            scores[proxy] = getOverlap(testWords, UDWORDS[proxy])
        PROXIES[UDdir] = sorted(scores, key=scores.get, reverse=True)[0]
 
def pred(model, dev, output, datasetID):
    evalFile = output + '.eval'
    isEmpty = (not os.path.isfile(evalFile)) or (os.path.isfile(evalFile) and os.stat(evalFile).st_size == 0)
    if model != '' and isEmpty:
        cmd = ' '.join(['python3 predict.py', model, dev, output, '--device 0', '--dataset ' + datasetID])
        print(cmd)

outDir = 'preds/'
if not os.path.isdir(outDir):
    os.mkdir(outDir)

udPath = 'data/ud-treebanks-v2.7/'
for UDdir in os.listdir(udPath):
    train, dev, test = myutils.getTrainDevTest(udPath + UDdir)
    
    if dev == '' and myutils.hasColumn(test, 1):
        for config in ['concat', 'concat.smoothed', 'sepDec.smoothed', 'datasetEmbeds.smoothed']:
            name = 'fullUD' + config
            model = myutils.getModel(name)
            output = outDir + name + '.' + UDdir + '.test.conllu'
            datasetID = PROXIES[UDdir]
            pred(model, test, output, datasetID)
        model = myutils.getModel(PROXIES[UDdir])
        output = outDir + 'self.' + UDdir + '.dev.conllu'
        pred(model, test, output, datasetID)


