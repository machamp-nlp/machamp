import os
import sys
import torch

from transformers import AutoTokenizer
from machamp.model.machamp import MachampModel

model = torch.load(sys.argv[1])

outPath = sys.argv[2]
if not os.path.isdir(outPath):
    os.mkdir(outPath)
model.mlm.save_pretrained(outPath)
tokenizer = AutoTokenizer.from_pretrained(model.mlm.name_or_path)
tokenizer.save_pretrained(outPath)
config = model.mlm.config.to_json_file(outPath + '/config.json')
