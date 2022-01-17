import os
import myutils

UDWORDS = {}
EMPTIES = set()
for udPath in ['data/ud-treebanks-v' + myutils.UDversion + '.singleToken/', 'data/ud-treebanks-v2.extras.singleToken/']:
    for UDdir in os.listdir(udPath):
        if not (os.path.isdir(udPath + UDdir) and UDdir.startswith('UD')):
            continue
        train, dev, test = myutils.getTrainDevTest(udPath + UDdir)
        if not myutils.hasColumn(test, 1):
            #print('NOWORDS', test)
            continue
        if train == '':
            EMPTIES.add(udPath + UDdir)
        else:
            words = myutils.getWords(train)
            UDWORDS[udPath + UDdir] = set(words)

PROXIES = {}
for UDdir in EMPTIES: 
    _, _, test = myutils.getTrainDevTest(UDdir)
    testWords = myutils.getWords(test, 10)
    scores = {}
    for proxy in UDWORDS:
        scores[proxy] = myutils.getOverlap(testWords, UDWORDS[proxy])
    PROXIES[UDdir] = sorted(scores, key=scores.get, reverse=True)[0]

def pred(model,test, output, datasetID):
    evalFile = output + '.eval'
    isEmpty = (not os.path.isfile(output)) or (os.path.isfile(output) and os.stat(output).st_size == 0)
    #print(output, isEmpty, model!= '')
    if model != '' and isEmpty:
        cmd = ' '.join(['python3 predict.py', model, test, output, '--dataset ' + datasetID])
        print(cmd)

outDir = 'preds' + myutils.UDversion + '/'
if not os.path.isdir(outDir):
    os.mkdir(outDir)

for UDdir in list(UDWORDS) + list(EMPTIES):
    for seed in myutils.seeds:
        train, dev, test = myutils.getTrainDevTest(UDdir)
    
        datasetName = UDdir.split('/')[-1]
        datasetID = UDdir if train != '' else PROXIES[UDdir]
        datasetID = datasetID.split('/')[-1]

        for config in ['concat', 'concat.smoothed', 'sepDec.smoothed', 'datasetEmbeds.smoothed']:
            name = 'fullUD' + config + '.' + str(seed)
            model = myutils.getModel(name)
            output = outDir + name + '.' + datasetName + '.test.' + seed + '.conllu'
            pred(model, test, output, datasetID)

        model = myutils.getModel(datasetID + '.' + str(seed))
        output = outDir + 'self.' + datasetName + '.test.' + seed + '.conllu'
        pred(model, test, output, datasetID)


