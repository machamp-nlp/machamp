import os
import myutils

UDWORDS = {}
for udPath in ['data/ud-treebanks-v2.7/', 'data/ud-treebanks-v2.7.extras/']:
    for UDdir in os.listdir(udPath):
        if not (os.path.isdir(udPath + UDdir) and UDdir.startswith('UD')):
            continue
        train, dev, test = myutils.getTrainDevTest(udPath + UDdir)
        if not myutils.hasColumn(test, 1):
            #print('NOWORDS', test)
            continue
        words = myutils.getWords(train)
        UDWORDS[udPath + UDdir] = set(words)

PROXIES = {}
for UDdir in UDWORDS: 
    train, dev, test = myutils.getTrainDevTest(UDdir)
    if train == '':
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
        #print((not os.path.isfile(output)), (os.path.isfile(output) and os.stat(output).st_size == 0), cmd)
        print(cmd)
    #else:
    #    print('ERROR', model, isEmpty, output)

outDir = 'preds/'
if not os.path.isdir(outDir):
    os.mkdir(outDir)

for UDdir in UDWORDS:
    for seed in myutils.seeds:
        train, dev, test = myutils.getTrainDevTest(UDdir)
    
        datasetName = UDdir.split('/')[-1]
        datasetID = UDdir if train != '' else PROXIES[UDdir]
        datasetID = datasetID.split('/')[-1]

        for config in ['concat', 'concat.smoothed', 'sepDec.smoothed', 'datasetEmbeds.smoothed']:
            name = 'ALLfullUD' + config + '.' + str(seed)
            model = myutils.getModel(name)
            output = outDir + name + '.' + datasetName + '.test.' + seed + '.conllu'
            pred(model, test, output, datasetID)

        model = myutils.getModel(datasetID + '.' + str(seed))
        output = outDir + 'self.' + datasetName + '.test.' + seed + '.conllu'
        pred(model, test, output, datasetID)


