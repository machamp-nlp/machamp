"""
Training script useful for debugging UDify and AllenNLP code
"""

import os
import copy
import datetime
import logging
import argparse

from allennlp.common import Params
from allennlp.common.util import import_submodules
from allennlp.commands.train import train_model

from machamp import util

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--name", default="", type=str, help="Log dir name")
parser.add_argument("--dataset_config", default="", type=str, help="Configuration file for datasets")
parser.add_argument("--parameters_config", default="configs/params.json", type=str, help="Configuration file for parameters of the model")
parser.add_argument("--device", default=None, type=int, help="CUDA device; set to -1 for CPU")
parser.add_argument("--resume", type=str, help="Resume training with the given model")
parser.add_argument("--archive_bert", action="store_true", help="Archives the finetuned BERT model after training")

args = parser.parse_args()

log_dir_name = args.name
if not log_dir_name:
    file_name = args.dataset_config if args.dataset_config else args.parameters_config
    log_dir_name = os.path.basename(file_name).split(".")[0]

if not args.resume:
    serialization_dir = os.path.join("logs", log_dir_name, datetime.datetime.now().strftime("%Y.%m.%d_%H.%M.%S"))

    overrides = {}
    if args.device is not None:
        overrides["trainer"] = {"cuda_device": args.device}
    train_params = util.merge_configs(args.parameters_config, args.dataset_config, overrides)
else:
    serialization_dir = args.resume
    train_params = Params.from_file(os.path.join(serialization_dir, "config.json"))
    if args.device is not None:
        train_params["trainer"]["cuda_device"] =  args.device
    train_params.to_file(os.path.join(serialization_dir, 'config.json'))


predict_params = train_params.duplicate()
import_submodules("machamp")

train_model(train_params, serialization_dir, recover=bool(args.resume))

for dataset in predict_params['dataset_reader']['datasets']:
    dataset_params = predict_params.duplicate()
    dev_file = dataset_params['dataset_reader']['datasets'][dataset]['dev']
    dev_pred = os.path.join(serialization_dir, dataset + '.dev.out')
    dev_eval = os.path.join(serialization_dir, dataset + '.dev_results.json')
    datasets = copy.deepcopy(dataset_params['dataset_reader']['datasets'])
    for iterDataset in datasets:
        if iterDataset != dataset:
            del dataset_params['dataset_reader']['datasets'][iterDataset]
    
    util.predict_model("machamp_predictor", dataset_params, serialization_dir, dev_file, dev_pred)

if args.archive_bert:
    #TODO fix hardcoded path?
    bert_config = "config/archive/bert-base-multilingual-cased/bert_config.json"
    util.archive_bert_model(serialization_dir, bert_config)

# If we want to use trainer>num_serialized_models_to_keep we need to comment this automatic cleanup
util.cleanup_training(serialization_dir, keep_archive=True)
