from allennlp.common import Params
import random
random.seed(8446)



cmds = []
base = Params.from_file('configs/params.json')
for dropout in [.3,.5,.7]: #.5
    for bertDropout in [.1,.2]:#.1
        for maskProb in [.1,.2,.3]:#.1
            for learnRate, batchSize in [(1e-4, 16), (1e-3, 32), (1e-2, 64)]:#32
                for decoderDropout in [.1,.3,.5]:#.3
                    # link decoderDropout to dropout?
                    base['model']['text_field_embedder']['dropout'] = dropout
                    base['model']['text_field_embedder']['token_embedders']['bert']['dropout'] = bertDropout
                    base['model']['word_dropout'] = maskProb
                    base['trainer']['optimizer']['lr'] = learnRate
                    base['iterator']['batch_size'] = batchSize
                    base['iterator']['maximum_samples_per_batch'] = ['num_tokens', batchSize * 100]
                    base['model']['default_decoder']['dropout'] = decoderDropout

                    # save config
                    name = '.'.join([str(x).replace('.','') for x in [dropout, bertDropout, maskProb, batchSize, decoderDropout]])
                    base.to_file('configs/' + name + '.json')
                    for dataset in ['glue', 'pmb', 'ewt']:
                        cmd = 'python3 train.py --device 0 --name ' + dataset + '.' + name 
                        cmd += ' --dataset_config configs/' + dataset + '.json'
                        cmd += ' --parameters_config configs/' + name + '.json'
                        cmds.append(cmd)

random.shuffle(cmds)
for cmd in cmds:
    print(cmd)

