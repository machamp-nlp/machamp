import sys
import os

def mask(path):
    data = []
    for line in open(path):
        if line.startswith('# text'):
            continue
        tok = line.strip().split('\t')
        if len(tok) == 10:
            tok[1] = '_'
            tok[2] = '_'
            tok[9] = '_'
        data.append('\t'.join(tok))
    outFile = open(path, 'w')
    for line in data:
        outFile.write(line + '\n')
    outFile.close()
    
            

for dataset in ['UD_English-Dundee', 'UD_English-ESL', 'UD_English-GUMReddit', 'UD_French-ExtremeUGC0.6.2', 'UD_French-FTB', 'UD_Hindi_English-HIENCS', 'UD_Mbya_Guarani-Dooley']:
    for tgtFile in os.listdir(sys.argv[1]):
        if tgtFile.endswith('conllu') and dataset in tgtFile:
            mask(sys.argv[1] + '/' + tgtFile)

