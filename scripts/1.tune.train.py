from allennlp.common import Params
import random
random.seed(8446)
import myutils

cmds = []
for embed in ['mbert', 'rembert']:
    if embed == 'mbert':
        base = Params.from_file('configs/params.json')
        epochs = [15,20]
        batch_sizes = [16,32,64]
    if embed == 'rembert':
        base = Params.from_file('configs/params-rembert.json')
        epochs = [20]
        batch_sizes = [32]
    for epoch in epochs:
        for cut_frac in [.1,.2, .3]:
            for batch_size in batch_sizes:
                for learnRate in [1e-3, 1e-4, 1e-5]:
                    for dropout in [.1,.2,.3]:
                        base['trainer']['num_epochs'] = epoch
                        base['trainer']['optimizer']['lr'] = learnRate
                        base['trainer']['learning_rate_scheduler']['cut_frac'] = cut_frac
                        base['data_loader']['batch_sampler']['batch_size'] = batch_size
                        base['model']['dropout'] = dropout
                        name = '.'.join([str(x).replace('.','') for x in [embed, cut_frac, learnRate, dropout, batch_size, str(epoch)]])
                        base.to_file('configs/tmp/' + name + '.json')
                        for dataset in ['xtreme']:
                            cmd = 'python3 train.py --name ' + dataset + '.' + name 
                            cmd += ' --dataset_config configs/' + dataset + '.json'
                            cmd += ' --parameters_config configs/tmp/' + name + '.json'
                            if myutils.getModel(dataset + '.' + name) == '': 
                                cmds.append(cmd)

random.shuffle(cmds)
for cmd in cmds:
    print(cmd)


