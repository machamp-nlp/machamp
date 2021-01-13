import os
import myutils

outDir = 'preds/'
if not os.path.isdir(outDir):
    os.mkdir(outDir)

udPath = 'data/ud-treebanks-v2.7/'
for UDdir in os.listdir(udPath):
    _, gold, _ = myutils.getTrainDevTest('data/ud-treebanks-v2.7/' + UDdir)
    for config in ['concat', 'sepDec', 'datasetEmbeds']:
        for smoothing in [False, True]:
            devPred = outDir + 'fullUD' + config + ('.smoothed.' if smoothing else '.') + UDdir + '.dev.conllu'
            if os.path.isfile(devPred):
                cmd = 'python3 scripts/misc/conll18_ud_eval.py ' + gold + ' '+ devPred + ' > ' + devPred + '.eval'
                #print(cmd)
                os.system(cmd)
    devPred = outDir + 'self.' + UDdir + '.dev.conllu'
    if os.path.isfile(devPred):
        cmd = 'python3 scripts/misc/conll18_ud_eval.py ' + gold + ' '+ devPred + ' > ' + devPred + '.eval'
        #print(cmd)
        os.system(cmd)

    

