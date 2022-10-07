import argparse
import logging
import sys

import torch

from machamp.predictor.predict import predict2

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("torch_model", type=str, help="The path to the pytorch (*.pt) model.")
parser.add_argument("file_paths", nargs='+',
                    help="contains a list of input and output files. You can predict on multiple files by having a "
                         "structure like: input1 output1 input2 output2.")
parser.add_argument("--dataset", default=None, type=str,
                    help="name of the dataset, needed to know the word_idx/sent_idxs to read from")
parser.add_argument("--device", default=None, type=int, help="CUDA device number; set to -1 for CPU.")
parser.add_argument("--batch_size", default=32, type=int, help="The size of each prediction batch.")
parser.add_argument("--raw_text", action="store_true", help="Input raw sentences, one per line in the input file.")
parser.add_argument("--topn", default=None, type=int, help='Output the top-n labels and their probability.')
args = parser.parse_args()

logger.info('cmd: ' + ' '.join(sys.argv) + '\n')
if len(args.file_paths) % 2 == 1:
    logger.error('Error: the number of files passed is not even. You need to pass an output file for each input file: ' + str(
        args.file_paths))
    exit(1)

if args.device == None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
elif args.device == -1:
    device = 'cpu'
else:
    device = 'cuda:' + str(args.device)

logger.info('loading model...')
model = torch.load(args.torch_model, map_location=device)
model.device = device

if args.topn != None:
    for decoder in model.decoders:
        model.decoders[decoder].topn = args.topn

for dataIdx in range(0, len(args.file_paths), 2):
    input_path = args.file_paths[dataIdx]
    output_path = args.file_paths[dataIdx + 1]
    logger.info('predicting on ' + input_path + ', saving on ' + output_path)
    predict2(model, input_path, output_path, args.dataset, args.batch_size, args.raw_text, device)
