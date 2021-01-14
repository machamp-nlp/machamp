import os
import copy
import glob
import logging
import torch
import tarfile
import json
import numpy as np
from datetime import datetime

from transformers.configuration_utils import PretrainedConfig
from allennlp.data.tokenizers import PretrainedTransformerTokenizer
from allennlp.commands.train import train_model_from_file
from allennlp.common import Params
from allennlp.models.archival import load_archive
from allennlp.models import Model
from allennlp.commands.predict import _PredictManager

from machamp.predictor import MachampPredictor

logger = logging.getLogger(__name__)

def merge_configs(params_config_path, dataset_config_paths):
    params_config = Params.from_file(params_config_path)
    datasets_config = {}
    if type(dataset_config_paths) == list:
        for dataset_config_path in dataset_config_paths:
            datasets_config.update(Params.from_file(dataset_config_path).as_dict())
    else:
        datasets_config = Params.from_file(dataset_config_paths).as_dict()

    # to support reading from multiple files we add them to the datasetreader constructor instead
    # the following ones are there just here to make allennlp happy
    params_config['train_data_path'] = 'TRAINPLACEHOLDER'
    params_config['validation_data_path'] = 'DEVPLACEHOLDER'

    params_config['dataset_reader']['datasets'] = datasets_config

    ordered_stuff = {}
    new_decoders = {}
    for dataset in datasets_config:
        for task in datasets_config[dataset]['tasks']:
            # start out with default decoder
            task_decoder = copy.deepcopy(params_config['model']['decoders']['default'].as_dict())
            task_decoder['dataset_embeds_dim'] = params_config['model']['dataset_embeds_dim']

            # add task_type defaults
            task_type = datasets_config[dataset]['tasks'][task]['task_type']
            if task_type not in params_config['model']['decoders']:
                tasks_list = [task_str for task_str in params_config['modeĺ']['decoders']]
                del tasks_list['default']
                logger.error('Task type ' + task_type + " is not supported, please use one of " + str(tasks_list))
            task_decoder.update(params_config['model']['decoders'][task_type].as_dict())

            # add anything that is defined in dataset_config
            task_decoder.update(datasets_config[dataset]['tasks'][task])

            # add name of task to task itself (used to log metrics)
            task_decoder['task'] = task

            # Used to create an ordered list later
            ordered_stuff[task] = [task_decoder['order'], task_type]

            # remove items only used in datareader, and items save in ordered_stuff
            for item in ['column_idx', 'task_type', 'order']:
                if item in task_decoder:
                    del task_decoder[item]
            new_decoders[task] = task_decoder 

        if 'max_sents' not in datasets_config[dataset] and params_config['model']['default_max_sents'] != 0:
            params_config['dataset_reader']['datasets'][dataset]['max_sents'] = params_config['model']['default_max_sents']
    if 'default_max_sents' in params_config['model']:
        del params_config['model']['default_max_sents']

    params_config['model']['decoders'] = new_decoders

    # Used in the machamp model to decide which order to use
    # generate ordered lists, which make it easier to use in the machamp model
    ordered_tasks = []
    ordered_task_types = []
    no_padding = []
    for label, idx in sorted(ordered_stuff.items(), key=lambda item: item[1]):
        ordered_tasks.append(label)
        ordered_task_types.append(ordered_stuff[label][1])
        #if ordered_stuff[label][1] == 'dependency':
        #    no_padding.append(label + '_rels')
        #    no_padding.append(label + '_head_indices')
        #else:
        #    no_padding.append(label)
        ##seq2seq is not included
    params_config['model']['tasks'] = ordered_tasks
    params_config['model']['task_types'] = ordered_task_types
    #params_config['vocabulary'] = {'non_padded_namespaces': ['dataset']}
    #params_config['vocabulary'] = {'non_padded_namespaces': no_padding + ['dataset', 'src_tokens']}

    return params_config


def walk_and_replace_dict(data, orig, new):
    for item in data:
        if type(data[item]) == dict:
            data[item] = walk_and_replace_dict(data[item], orig, new)
        if type(data[item]) == Params:
            data[item] = walk_and_replace_dict(data.as_dict()[item], orig, new) 
        print(item)
        if data[item] == orig:
            data[item] = new
    return data

def train(config, name, resume, finetune):
    now = datetime.now()
    serialization_dir = 'logs/' + name + '/' + now.strftime("%Y.%m.%d_%H.%M.%S") + '/'
    if resume:
        serialization_dir = name
    if not os.path.isdir(serialization_dir):
        os.makedirs(serialization_dir)
    
    if finetune not in [None, '']:
        # prepare embeddings
        model_path = os.path.join(finetune, 'torch_model')
        archive_model(finetune, model_path)
        
        # update config
        model_name = config['model']['text_field_embedder']['token_embedders']['tokens']['model_name']
        config = walk_and_replace_dict(config, model_name, model_path)
        
    config_path = serialization_dir + 'config.json'
    config.to_file(config_path)
    
        

    model = train_model_from_file(config_path,
                        serialization_dir,
                        file_friendly_logging=True,
                        force=(not resume), 
                        recover=resume)
    if os.path.isfile(serialization_dir + 'vocabulary/.lock'):
        os.remove(serialization_dir + 'vocabulary/.lock')
    return model, serialization_dir

def predict_model_with_archive(predictor: str, params: Params, archive: str,
                               input_file: str, output_file: str, batch_size: int = None):
    task_types = []
    for dataset in params['dataset_reader']['datasets']:
        for task in params['dataset_reader']['datasets'][dataset]['tasks']:
            task_types.append(params['dataset_reader']['datasets'][dataset]['tasks'][task]['task_type'])
    if 'mlm' in task_types:
        logger.warning("No prediction is written, as it is unclear what to output when predicting on dev/test data with MLM")
        return


    if 'cuda_device' in params['trainer']:
        if params["trainer"]["cuda_device"] != -1:
            archive = load_archive(archive, cuda_device=params["trainer"]["cuda_device"])
        else:
            archive = load_archive(archive)
    elif torch.cuda.is_available():
        archive = load_archive(archive, cuda_device=0)
    else:
        archive = load_archive(archive)


    for item in archive.config.duplicate():
        archive.config.__delitem__(item)
    for item in params:
        archive.config[item] = params.as_dict()[item]

    archive.validation_dataset_reader.datasets = params['dataset_reader']['datasets']

    predictor = MachampPredictor.from_archive(archive, predictor)

    if batch_size == None:
        batch_size = params['data_loader']['batch_sampler']['batch_size']

    manager = _PredictManager(predictor,
                              input_file,
                              output_file,
                              batch_size,
                              print_to_console=False,
                              has_dataset_reader=True)
    manager.run()


def archive_model(serialization_dir: str, out_dir: str):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)
    # get pytorch model
    weights_file = os.path.join(serialization_dir, 'best.th')
    if not os.path.isfile(weights_file):
        # extract from model.tar.gz
        tar = tarfile.open(os.path.join(serialization_dir, 'model.tar.gz'), "r:gz")
        tar.extract('weights.th', serialization_dir)
        os.rename(os.path.join(serialization_dir, 'weights.th'), weights_file)
    model = torch.load(weights_file)
    bin_file = os.path.join(out_dir, "pytorch_model.bin")
    if not os.path.isfile(bin_file):
        torch.save(model, bin_file)

    # Get embeddings configuration
    model_config = Params.from_file(os.path.join(serialization_dir, 'config.json'))
    exact_model = model_config['model']['text_field_embedder']['token_embedders']['tokens']['model_name']
    config = PretrainedConfig.from_pretrained(exact_model)
    model_type = config.model_type
    config = config.to_dict()
    config['model_type'] = model_type
    json.dump(config, open(os.path.join(out_dir, 'config.json'), 'w'))

    # get tokenizer
    tokenizer1 = PretrainedTransformerTokenizer(exact_model)
    tokenizer1.tokenizer.save_pretrained(out_dir)

def clean_th_files(serialization_dir):
    model_path = os.path.join(serialization_dir, 'model.tar.gz')
    tar = tarfile.open(model_path, "r:gz")
    for check_file in ['config.json', 'weights.th', 'vocabulary']:
        if check_file not in tar.getnames():
            logger.warning("Did not clean up th files, because " + check_file + " is not found in " + model_path)
            return
    for thFile in os.listdir(serialization_dir):
        if thFile.endswith('.th'):
            os.remove(os.path.join(serialization_dir, thFile))

