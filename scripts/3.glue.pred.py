import os
from allennlp.common import Params
import myutils

predDir = 'preds/'
if not os.path.isdir(predDir):
    os.mkdir(predDir)
        
def pred(model, gold, out, task):
    if model != '' and not os.path.isfile(out):
        cmd = ' '.join(['python3 predict.py', model, gold, out, '--dataset', task])
        print(cmd)

glue = Params.from_file('configs/glue.json')
for task in glue:
    gold = glue[task]['validation_data_path']
    for setting in ['glue', 'glue.smoothSampling']:
        model = myutils.getModel(setting)
        out = predDir + setting + '.' + task
        pred(model, gold, out, task)

    # single runs
    model = myutils.getModel('glue.' + task)
    out = predDir + 'glue.single.' + task
    pred(model, gold, out, task)

