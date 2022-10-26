import argparse
import sys

import torch

from machamp.model import trainer

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_configs", nargs='+',
                    help="Path(s) to dataset configurations (use --sequential to train on them sequentially, "
                         "default is joint training).")
parser.add_argument("--name", default="", type=str, help="Log dir name.")
parser.add_argument("--sequential", action="store_true",
                    help="Enables finetuning sequentially, this will train the same weights once for each "
                         "dataset_config you pass.")
parser.add_argument("--parameters_config", default="configs/params.json", type=str,
                    help="Configuration file for parameters of the model.")
parser.add_argument("--device", default=None, type=int, help="CUDA device; set to -1 for CPU.")
parser.add_argument("--resume", default='', type=str,
                    help='Finalize training on a model for which training abruptly stopped. Give the path to the log '
                         'directory of the model.')
parser.add_argument("--retrain", type=str, default='',
                    help="Retrain on an previously train MaChAmp model. Specify the path to model.tar.gz and add a "
                         "dataset_config that specifies the new training.")
parser.add_argument("--seed", type=int, default=8446, help="seed to use for training.")
args = parser.parse_args()

if args.resume == '' and (args.dataset_configs == None or len(args.dataset_configs) == 0):
    print('Please provide at least 1 dataset configuration')
    exit(1)

if args.device == None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
elif args.device == -1:
    device = 'cpu'
else:
    device = 'cuda:' + str(args.device)

name = args.name
if args.resume == '' and name == '':
    names = [name[name.rfind('/') + 1: name.rfind('.') if '.' in name else len(name)] for name in args.dataset_configs]
    name = '.'.join(names)
if args.resume != '':
    name = args.resume.split('/')[1]

cmd = ' '.join(sys.argv)

if args.sequential:
    prevDir = trainer.train(name + '.0', args.resume, args.dataset_configs[0], device, args.parameters_config,
                            args.retrain, args.seed, ' '.join(sys.argv))
    for datasetIdx, dataset in enumerate(args.dataset_configs[1:]):
        modelName = name + '.' + str(datasetIdx + 1)
        prevDir = trainer.train(modelName, args.parameters_config, dataset, device, None, prevDir, args.seed, cmd)

else:
    trainer.train(name, args.parameters_config, args.dataset_configs, device, args.resume, args.retrain, args.seed, cmd)
