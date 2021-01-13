import os
import myutils
outDir = 'preds/'

udPath = 'data/ud-treebanks-v2.7/'
for UDdir in os.listdir(udPath):
    train, dev, test = myutils.getTrainDevTest(udPath + UDdir)
    
    if dev == '' and myutils.hasColumn(test, 1):
        for config in ['concat', 'concat.smoothed', 'sepDec.smoothed', 'datasetEmbeds.smoothed']:
            name = 'fullUD' + config
            devPred = outDir + name + '.' + UDdir + '.test.conllu'
            if os.path.isfile(devPred):
                cmd = 'python3 scripts/misc/conll18_ud_eval.py ' + test + ' '+ devPred + ' > ' + devPred + '.eval'
                os.system(cmd)
        devPred = outDir + 'self.' + UDdir + '.dev.conllu'
        if os.path.isfile(devPred):
            cmd = 'python3 scripts/misc/conll18_ud_eval.py ' + test + ' '+ devPred + ' > ' + devPred + '.eval'
            os.system(cmd)


