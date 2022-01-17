import os
import logging
import argparse
import tarfile
import copy
from pathlib import Path

from allennlp.common import Params
from allennlp.common.util import import_module_and_submodules

from machamp import util

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("archive", type=str, help="The archive file")
parser.add_argument("input_file", type=str, help="The input file to predict")
parser.add_argument("pred_file", type=str, help="The output prediction file")
parser.add_argument("--dataset", default=None, type=str,
                    help="name of the dataset, needed to know the word_idx/sent_idxs to read from")
parser.add_argument("--device", default=None, type=int, help="CUDA device number; set to -1 for CPU")
parser.add_argument("--batch_size", default=None, type=int, help="The size of each prediction batch")
parser.add_argument("--raw_text", action="store_true", help="Input raw sentences, one per line in the input file.")
args = parser.parse_args()

import_module_and_submodules("machamp")

archive_dir = Path(args.archive).resolve().parent

if not os.path.isfile(archive_dir / "weights.th"):
    with tarfile.open(args.archive) as tar:
        tar.extractall(archive_dir)

config_file = archive_dir / "config.json"

params = Params.from_file(config_file)

if args.device is not None:
    params['trainer']['cuda_device'] = args.device
params['dataset_reader']['is_raw'] = args.raw_text

if args.dataset is None and len(params['dataset_reader']['datasets']) > 1:
    logger.error("please provide --dataset, because we currently don't support writing " +
                 "tasks of multiple datasets in one run.\nOptions: " +
                 str([dataset for dataset in params['dataset_reader']['datasets']]))
    exit(1)

if args.dataset not in params['dataset_reader']['datasets'] and args.dataset is not None:
    logger.error("Non existing --dataset option specified, please pick one from: " + 
                 str([dataset for dataset in params['dataset_reader']['datasets']]))
    exit(1)

if args.dataset:
    datasets = copy.deepcopy(params['dataset_reader']['datasets'])
    for iter_dataset in datasets:
        if iter_dataset != args.dataset:
            del params['dataset_reader']['datasets'][iter_dataset]

util.predict_model_with_archive("machamp_predictor", params, archive_dir, args.input_file, args.pred_file,
                                batch_size=args.batch_size)
