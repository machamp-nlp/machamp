from allennlp.common import Params
import os

def train(name):
    cmd = 'python3 train.py --dataset_config configs/tmp/' + name + '.json'
    print(cmd)

for udPath in ['data/ud-treebanks-v2.9.singleToken/', 'data/ud-treebanks-v2.extras.singleToken/']:
    for treebank in os.listdir(udPath):
        if os.path.isdir(udPath + '/' + treebank):
            train(treebank)

glue = Params.from_file('configs/glue.json')
for task in glue:
    train('glue.' + task)

