import os
from allennlp.common import Params
import myutils

udPath = 'data/ud-treebanks-v2.7/'
UDSETS = []
for UDdir in os.listdir(udPath):
    if not UDdir.startswith("UD"):
        continue
    _, _, test = myutils.getTrainDevTest(udPath + UDdir)
    if test != '' and myutils.hasColumn(test, 1):
        UDSETS.append(UDdir)


outDir = 'preds/'
for setting in ['self', 'concat.smoothed', 'sepDec.smoothed', 'datasetEmbeds.smoothed', 'concat']:
    scores = []
    for UDdir in UDSETS:
        scores = []
        for seed in myutils.seeds:
            if setting == 'self':
                output = 'preds/self.' + UDdir
            else:
                output = 'preds/ALLfullUD' + setting + '.' + str(seed) + '.' + UDdir 
            output = output + '.test.' + str(seed) + '.conllu.eval'
            if os.path.isfile(output) and os.stat(output).st_size != 0 and os.stat(output).st_size < 100:
                scores.append(float(open(output).readline().strip().split()[-1]))
            else:
                scores.append(0.0)
                print("NF", output)
        scores.append(sum(scores)/len(scores))
    print(setting + ' & ' + str(round(sum(scores)/len(scores), 2)))
            
print()
glue = Params.from_file('configs/glue.json')
for setting in ['glue', 'glue.smoothSampling', 'glue.single']:
    scores = []
    for task in glue:
        if task == 'wnli':
            continue
        output = outDir + setting + '.' + task + '.eval'
        if os.path.isfile(output):
            score = float(open(output).readline().split()[0])
        else:
            score = 0.0
        scores.append(score)
    print(setting + ' & ' + str(round(sum(100 * scores)/len(scores), 2)))


