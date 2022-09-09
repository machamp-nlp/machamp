import argparse
import logging
import sys

import torch

from machamp.predictor.predict import predict2

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    level=logging.INFO, handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("archive", type=str, help="The path to the model.")
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

print('cmd: ' + ' '.join(sys.argv))
print()
if len(args.file_paths) % 2 == 1:
    print('Error: the number of files passed is not even. You need to pass an output file for each input file: ' + str(
        args.file_paths))

if args.device == None:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
elif args.device == -1:
    device = 'cpu'
else:
    device = 'cuda:' + str(args.device)

print('loading model...')
model = torch.load(args.archive, map_location=device)
if args.topn != None:
    for decoder in model.decoders:
        model.decoders[decoder].topn = args.topn

for dataIdx in range(0, len(args.file_paths), 2):
    input_path = args.file_paths[dataIdx]
    output_path = args.file_paths[dataIdx + 1]
    print('predicting on ' + input_path + ', saving on ' + output_path)
    predict2(model, input_path, output_path, args.dataset, args.batch_size, args.raw_text, device)
