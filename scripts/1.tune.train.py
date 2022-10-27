import myutils
import os
import json
import random

lms = ['bert-base-multilingual-cased', 'xlm-roberta-large']

cmds = []
defaultPath = 'configs/params.json'
for dataset in ['ewt']:
    for lm in lms:
        for lr in [0.00001, 0.000005, 0.000001]:
            for batch_size in [16,32]:
                for gradual in [False, True]:
                    for disc in [False, True]:
                        name = '.'.join([str(x) for x in [dataset, lm.replace('/', '_'), lr, batch_size, gradual, disc]])
                        if myutils.getModel(name) != '':
                            continue
                        config = myutils.load_json(defaultPath)
                        config['transformer_model'] = lm
                        config['training']['optimizer']['lr'] = lr
                        config['batching']['batch_size'] = batch_size
                        config['training']['learning_rate_scheduler']['gradual_unfreezing'] = gradual
                        config['training']['learning_rate_scheduler']['discriminative_fine_tuning'] = disc
                        tgtPath = 'configs/params.' + name + '.json'
                        json.dump(config, open(tgtPath, 'w'), indent=4)
                        cmd = 'python3 train.py --dataset_config configs/' + dataset + '.json --parameters_config ' + tgtPath + ' --name ' + name
                        cmds.append(cmd)

random.shuffle(cmds)
for cmd in cmds:
    print(cmd)

