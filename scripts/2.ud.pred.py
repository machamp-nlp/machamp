import os
import myutils
 
def pred(model, dev, output):
    evalFile = output + '.eval'
    isEmpty = (not os.path.isfile(evalFile)) or (os.path.isfile(evalFile) and os.stat(evalFile).st_size == 0)
    if model != '' and isEmpty:
        cmd = ' '.join(['python3 predict.py', model, dev, output, '--device 0', '--dataset ' + UDdir])
        print(cmd)

outDir = 'preds/'
if not os.path.isdir(outDir):
    os.mkdir(outDir)

udPath = 'data/ud-treebanks-v2.7/'
for UDdir in os.listdir(udPath):
    train, dev, test = myutils.getTrainDevTest(udPath + UDdir)
    
    if dev != '' and myutils.hasColumn(train, 1):
        for config in ['concat', 'sepDec', 'datasetEmbeds']:
            for smoothing in [False, True]:
                name = 'fullUD' + config + ('.smoothed' if smoothing else '')
                model = myutils.getModel(name)
                output = outDir + name + '.' + UDdir + '.dev.conllu'
                pred(model, dev, output)
        # single treebank runs
        model = myutils.getModel(UDdir)
        output = outDir + 'self.' + UDdir + '.dev.conllu'
        pred(model, dev, output)

