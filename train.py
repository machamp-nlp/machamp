import argparse
import copy
import logging
import os

from allennlp.common import Params
from allennlp.common.util import import_module_and_submodules

from machamp import util

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="", type=str, help="Log dir name")
parser.add_argument("--dataset_config", default="", type=str, help="Configuration file for datasets")
parser.add_argument("--dataset_configs", default=[], nargs='+', help="If you want to train on multiple datasets simultaneously (use --sequential to train on them sequentially)")
parser.add_argument("--sequential", action="store_true", help="Enables finetuning sequentially, this will train the same weights once for each dataset_config you pass")
parser.add_argument("--parameters_config", default="configs/params.json", type=str,
                    help="Configuration file for parameters of the model")
parser.add_argument("--device", default=None, type=int, help="CUDA device; set to -1 for CPU")
parser.add_argument("--resume", default='', type=str, help="Finalize training on a model for which training abrubptly stopped. Give the path to the log directory of the model.")
parser.add_argument("--finetune", type=str, default='', help="Retrain on an previously train MaChAmp AllenNLP model. Specify the path to model.tar.gz and add a dataset_config that specifies the new training.")
parser.add_argument("--seed", type=int, default=-1, help="seed to use for training") #TODO

args = parser.parse_args()

if args.dataset_config == '' and args.resume in ['', None] and args.dataset_configs == []:
    logger.error('when not using --resume, specifying at least --dataset_config is required')
    exit(1)

if args.dataset_configs == []:
    args.dataset_configs.append(args.dataset_config)

import_module_and_submodules("machamp")

def train(name, resume, dataset_configs, device, parameters_config, finetune):
    if resume:
        train_params = Params.from_file(resume + '/config.json')
    else:
        train_params = util.merge_configs(parameters_config, dataset_configs, args.seed)

    if device is not None:
        train_params['trainer']['cuda_device'] = device
        # the config will be read twice, so for --resume we want to overwrite the config file
        if resume:
            train_params.to_file(resume + '/config.json')

    if resume:
        name = resume

    model, serialization_dir = util.train(train_params, name, resume, finetune)
    
    # now loads again for every dataset, = suboptimal
    # alternative would be to load the model once, but then the datasetReader has 
    # to be adapted for each dataset!
    #del train_params['dataset_reader']['type']
    #reader = MachampUniversalReader(**train_params['dataset_reader'])
    #predictor = MachampPredictor(model, reader)

    for dataset in train_params['dataset_reader']['datasets']:
        dataset_params = train_params.duplicate().as_dict()
        if 'validation_data_path' not in dataset_params['dataset_reader']['datasets'][dataset]:
            continue
        dev_file = dataset_params['dataset_reader']['datasets'][dataset]['validation_data_path']
        dev_pred = os.path.join(serialization_dir, dataset + '.dev.out')
        datasets = copy.deepcopy(dataset_params['dataset_reader']['datasets'])
        for iter_dataset in datasets:
            if iter_dataset != dataset:
                del dataset_params['dataset_reader']['datasets'][iter_dataset]

        util.predict_model_with_archive("machamp_predictor", dataset_params,
                                        serialization_dir + '/model.tar.gz', dev_file, dev_pred)
    
    util.clean_th_files(serialization_dir)
    return serialization_dir

name = args.name
if name == '':
    names = [name[name.rfind('/')+1: name.rfind('.') if '.' in name else len(name)] for name in args.dataset_configs]
    name = '.'.join(names)

if args.sequential:
    oldDir = train(name + '.0', args.resume, args.dataset_configs[0], args.device, args.parameters_config, args.finetune)
    for datasetIdx, dataset in enumerate(args.dataset_configs[1:]):
        oldDir = train(name + '.' + str(datasetIdx+1), False, dataset, args.device, args.parameters_config, oldDir)
else:
    train(name, args.resume, args.dataset_configs, args.device, args.parameters_config, args.finetune)

