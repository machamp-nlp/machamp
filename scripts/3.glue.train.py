import os
from allennlp.common import Params

if not os.path.isdir('configs/tmp/'):
    os.mkdir('configs/tmp/')

glue = Params.from_file('configs/glue.json')
for task in glue:
    taskConfig = Params({task: glue[task].as_dict()})
    jsonPath = 'configs/tmp/glue.' + task + '.json'
    taskConfig.to_file(jsonPath)
    cmd = 'python3 train.py --dataset_config ' + jsonPath
    print(cmd)

cmd = 'python3 train.py --dataset_config configs/glue.json'
print(cmd)

cmd = 'python3 train.py --dataset_config configs/glue.json --parameters_config configs/params.smoothSampling.json --name glue.smoothSampling'
print(cmd)

    

