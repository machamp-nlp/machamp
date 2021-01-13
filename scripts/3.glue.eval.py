import os
from allennlp.common import Params

predDir = 'preds/'
if not os.path.isdir(predDir):
    os.mkdir(predDir)
        
def eval(pred, gold, out):
    if os.path.isfile(pred):
        cor = 0
        total = 0
        for predLine, goldLine in zip(open(pred), open(gold)):
            predTok = predLine.strip().split('\t')
            goldTok = goldLine.strip().split('\t')
            if predTok[-1] == goldTok[-1]:
                cor += 1
            total += 1
        outFile = open(out, 'w')
        outFile.write(str(cor/total) + '\t' + str(cor) + '/' + str(total) + '\n')
        outFile.close()


glue = Params.from_file('configs/glue.json')
for task in glue:
    gold = glue[task]['validation_data_path']
    for setting in ['glue', 'glue.smoothSampling', 'glue.single']:
        pred = predDir + setting + '.' + task
        out = predDir + setting + '.' + task + '.eval'
        eval(pred, gold, out)


