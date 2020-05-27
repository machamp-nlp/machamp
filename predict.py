"""
Predict conllu files given a trained model
"""

import os
import shutil
import logging
import argparse
import tarfile
import copy
from pathlib import Path

from allennlp.common import Params
from allennlp.common.util import import_submodules
from allennlp.models.archival import archive_model

from machamp import util

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("archive", type=str, help="The archive file")
parser.add_argument("input_file", type=str, help="The input file to predict")
parser.add_argument("pred_file", type=str, help="The output prediction file")
parser.add_argument("--dataset", default=None, type=str, help="name of the dataset, needed to know the word_idx/sent_idxs to read from")
parser.add_argument("--eval_file", default=None, type=str,
                    help="If set, evaluate the prediction and store it in the given file")
parser.add_argument("--device", default=0, type=int, help="CUDA device number; set to -1 for CPU")
parser.add_argument("--batch_size", default=1, type=int, help="The size of each prediction batch")
parser.add_argument("--copy_other_columns", default=False, type=bool, help="Enable the copying of all columns not handled as a task or input data; if not enabled every other piece of data is replaced by an '_'")
parser.add_argument("--raw_text", action="store_true", help="Input raw sentences, one per line in the input file.")

args = parser.parse_args()

import_submodules("machamp")

archive_dir = Path(args.archive).resolve().parent

if not os.path.isfile(archive_dir / "weights.th"):
    with tarfile.open(args.archive) as tar:
        tar.extractall(archive_dir)

config_file = archive_dir / "config.json"

overrides = {}
if args.device is not None:
    overrides["trainer"] = {"cuda_device": args.device}
if args.copy_other_columns is not None:
    overrides["model"]= {"default_dataset":{"copy_other_columns": True}}
configs = [Params(overrides), Params.from_file(config_file)]
params = Params.from_file(config_file)
params['trainer']['cuda_device'] = args.device
predictor = "machamp_predictor"

if args.dataset:
    datasets = copy.deepcopy(params['dataset_reader']['datasets'])
    for iterDataset in datasets:
        if iterDataset != args.dataset:
            del params['dataset_reader']['datasets'][iterDataset]
params['dataset_reader']['isRaw'] = args.raw_text



util.predict_model_with_archive(predictor, params, archive_dir, args.input_file, args.pred_file,
                                    batch_size=args.batch_size)
