import os

idxs = {'CoLA':[1,-1],  'MNLI':[-1,8,9],  'MNLI-mis':[-1, 8, 9], 'MRPC':[0,3,4],  'QNLI':[3,1,2],  'QQP':[5,3,4],  'RTE':[3,1,2],  'SNLI':[0,5,6],  'SST-2':[1,0],  'WNLI':[3,1,2]}
testIdxs = {'CoLA':[1],  'MNLI':[8,9],  'MNLI-mis':[8, 9], 'MRPC':[3,4],  'QNLI':[1,2],  'QQP':[1,2],  'RTE':[1,2],  'SNLI':[0,5,6],  'SST-2':[1],  'WNLI':[1,2]}

#STS-B skipped for now

if not os.path.isdir('data'):
    os.mkdir('data')
glueDir = 'data/glue/'
if not os.path.isdir(glueDir):
    os.mkdir(glueDir)

if not os.path.isdir('GLUE-baselines'):
    os.system('git clone https://github.com/nyu-mll/GLUE-baselines.git')
    os.system('cp scripts/download_glue_data.py GLUE-baselines')
if not os.path.isdir('GLUE-baselines/glue_data'):
    os.system('cd GLUE-baselines && git clone https://github.com/wasiahmad/paraphrase_identification.git && python download_glue_data.py --data_dir glue_data --tasks all --path_to_mrpc=paraphrase_identification/dataset/msr-paraphrase-corpus && cd ..' )
    os.system('wget https://nlp.stanford.edu/projects/snli/snli_1.0.zip -O GLUE-baselines/SNLI.zip')
    os.system('unzip GLUE-baselines/SNLI.zip -d GLUE-baselines/')
    os.mkdir('GLUE-baselines/glue_data/SNLI/')
    os.system('cp GLUE-baselines/snli_1.0/snli_1.0_train.txt GLUE-baselines/glue_data/SNLI/train.tsv')
    os.system('cp GLUE-baselines/snli_1.0/snli_1.0_train.txt GLUE-baselines/glue_data/SNLI/dev.tsv')
    os.system('cp GLUE-baselines/snli_1.0/snli_1.0_train.txt GLUE-baselines/glue_data/SNLI/test.tsv')
    

for task in idxs:
    for split in ['train', 'dev', 'test']:
        readFirst = False
        if task == 'CoLA' and split != 'test':
            readFirst = True
        readFile = 'GLUE-baselines/glue_data/'+task+'/'+split +'.tsv'
        if task == 'MNLI' and split in ['dev', 'test']:
            readFile = 'GLUE-baselines/glue_data/MNLI/'+split +'_matched.tsv'
        elif task == 'MNLI-mis':
            if split in ['dev', 'test']:
                readFile = 'GLUE-baselines/glue_data/MNLI/'+split +'_mismatched.tsv'
            else:
                readFile = 'GLUE-baselines/glue_data/MNLI/'+split +'.tsv'

        if not os.path.isfile(readFile):
            print("error, " + readFile + " not found")
            continue
        
        outFile = open(glueDir + task + '.' + split, 'w')

        print(task, split, readFile)
        for lineIdx, line in enumerate(open(readFile)):
            if not readFirst:# or lineIdx > 20000:
                readFirst = True
                continue
            tok = line.strip().split('\t')
            if task == 'QQP' and len(tok) != 6 and split != 'test':
                continue
            
            if len(idxs[task]) == 2:
                if split == 'test':
                    outFile.write(tok[testIdxs[task][0]] + '\t_\n')
                else:
                    outFile.write(tok[idxs[task][1]] + '\t' + tok[idxs[task][0]] + '\n')
            else:
                if split == 'test':
                    outFile.write(tok[testIdxs[task][0]] + '\t' + tok[testIdxs[task][1]] + '\t_\n')
                else:
                    outFile.write(tok[idxs[task][1]] + '\t' + tok[idxs[task][2]] + '\t' + tok[idxs[task][0]] + '\n')
                
            #print(tok[goldIdx])
        outFile.close()
        
#os.system('rm -rf GLUE-baselines')
