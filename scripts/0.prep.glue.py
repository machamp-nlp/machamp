import os
from statistics import mode

if not os.path.isdir('data'):
    os.mkdir('data')

if not os.path.isdir('data/GLUE-baselines'):
    os.system('git clone https://github.com/nyu-mll/GLUE-baselines.git')
    os.system('cp scripts/download_glue_data.py GLUE-baselines')
if not os.path.isdir('data/GLUE-baselines/glue_data'):
    os.system('cd GLUE-baselines && git clone https://github.com/wasiahmad/paraphrase_identification.git && python3 download_glue_data.py --data_dir glue_data --tasks all --path_to_mrpc=paraphrase_identification/dataset/msr-paraphrase-corpus && cd ..' )
    os.system('wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip -O GLUE-baselines/SNLI.zip')
    os.system('unzip GLUE-baselines/SNLI.zip -d GLUE-baselines/')
    os.mkdir('GLUE-baselines/glue_data/SNLI/')
    os.system('cp GLUE-baselines/snli_1.0/snli_1.0_train.txt GLUE-baselines/glue_data/SNLI/train.tsv')
    os.system('cp GLUE-baselines/snli_1.0/snli_1.0_train.txt GLUE-baselines/glue_data/SNLI/dev.tsv')
    os.system('cp GLUE-baselines/snli_1.0/snli_1.0_train.txt GLUE-baselines/glue_data/SNLI/test.tsv')
    
    os.system('mv GLUE-baselines data')

for split in ['train', 'dev']:
    data = open('data/GLUE-baselines/glue_data/QQP/' + split + '.tsv').readlines()
    outFile = open('data/GLUE-baselines/glue_data/QQP/' + split + '.tsv', 'w')
    for line in data:
        if len(line.split('\t')) == 6:
            outFile.write(line)
    outFile.close()

    data = open('data/GLUE-baselines/glue_data/SNLI/' + split + '.tsv').readlines()
    outFile = open('data/GLUE-baselines/glue_data/SNLI/' + split + '.tsv', 'w')
    for line in data:
        tok = line.strip('\n').split('\t')
        if len(tok) != 11:
            mostVotes = mode(tok[-5:])
            tok[10] = mostVotes
        outFile.write('\t'.join(tok) + '\n')
    outFile.close()

