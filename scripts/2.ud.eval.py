import os
import myutils
outDir = 'preds' + myutils.UDversion + '/'

def isWrong(path):
    if not os.path.isfile(path):
        return True
    if os.stat(path).st_size < 40:
        return True
    if os.stat(path).st_size > 100:
        return True
    return False

for udPath in ['data/ud-treebanks-v' + myutils.UDversion + '/', 'data/ud-treebanks-v2.extras/']:
    for seed in myutils.seeds:
        for UDdir in os.listdir(udPath):
            if not (os.path.isdir(udPath + UDdir) and UDdir.startswith('UD')):
                continue
            train, dev, test = myutils.getTrainDevTest(udPath + UDdir)
            if UDdir == 'UD_Chukchi-HSE' or UDdir == 'UD_Marathi-UFAL':
                # these give errors in their original form, due to missing words (Marathi), or error in ud-conversion-tools
                test = test.replace('data/ud-treebanks-v' + myutils.UDversion + '.orig/', 'data/ud-treebanks-v' + myutils.UDversion + '/')

            if test != '' and myutils.hasColumn(test, 1):
                for config in ['concat', 'concat.smoothed', 'sepDec.smoothed', 'datasetEmbeds.smoothed']:
                    name = 'fullUD' + config + '.' + str(seed)
                    testPred = outDir + name + '.' + UDdir + '.test.' + str(seed) + '.conllu'
                    if os.path.isfile(testPred) and isWrong(testPred + '.eval'):
                        cmd = 'python3 scripts/misc/conll18_ud_eval.py -v ' + test + ' '+ testPred + ' > ' + testPred + '.eval'
                        print(cmd)
                        os.system(cmd)
                testPred = outDir + 'self.' + UDdir + '.test.' + str(seed) + '.conllu'
                if os.path.isfile(testPred) and isWrong(testPred + '.eval'):
                    cmd = 'python3 scripts/misc/conll18_ud_eval.py -v ' + test + ' '+ testPred + ' > ' + testPred + '.eval'
                    print(cmd)
                    os.system(cmd)
    
    
