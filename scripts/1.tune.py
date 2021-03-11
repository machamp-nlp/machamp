from allennlp.common import Params
import random
random.seed(8446)



cmds = []
base = Params.from_file('configs/params.json')
for cut_frac in [.1,.2, .3]:
    for learnRate in [1e-3, 1e-4, 1e-5]:
        for decay in [.35, .38, .5]:
            for dropout in [.1,.2,.2]:
                base['trainer']['optimizer']['lr'] = learnRate
                base['trainer']['learning_rate_scheduler']['decay_factor'] = decay
                base['trainer']['learning_rate_scheduler']['cut_frac'] = cut_frac
                base['model']['dropout'] = dropout
                name = '.'.join([str(x).replace('.','') for x in [cut_frac, learnRate, decay, dropout]])
                base.to_file('configs/' + name + '.json')
                for dataset in ['ewt', 'glue']:
                    cmd = 'python3 train.py --name ' + dataset + '.' + name 
                    cmd += ' --dataset_config configs/' + dataset + '.json'
                    cmd += ' --parameters_config configs/' + name + '.json'
                    cmds.append(cmd)

random.shuffle(cmds)
for cmd in cmds:
    print(cmd)


