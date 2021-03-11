import os
import myutils
from allennlp.common import Params

if not os.path.isdir('configs/tmp/'):
    os.mkdir('configs/tmp/')


    

for seed in myutils.seeds:
    glue = Params.from_file('configs/glue.json')
    for task in glue:
        taskConfig = Params({task: glue[task].as_dict()})
        jsonPath = 'configs/tmp/glue.' + task + '.json'
        taskConfig.to_file(jsonPath)
        cmd = 'python3 train.py --dataset_config ' + jsonPath + ' --name glue.' + task + '.' + seed + ' --seed ' + seed + ' --parameters_config configs/params.bert-large.json'
        print(cmd)

    cmd = 'python3 train.py --dataset_config configs/glue.json --name glue.' + seed + ' --seed ' + seed + ' --parameters_config configs/params.bert-large.json'
    print(cmd)

    cmd = 'python3 train.py --dataset_config configs/glue.json --parameters_config configs/params.bert-large.smoothSampling.json --name glue.smoothSampling.' + seed + ' --seed ' + seed
    print(cmd)

    

