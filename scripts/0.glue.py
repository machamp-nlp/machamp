import os

idxs = {'CoLA':[1,-1],  'MNLI':[-1,8,9],  'MNLI-mis':[-1, 8, 9], 'MRPC':[0,3,4],  'QNLI':[3,1,2],  'QQP':[5,3,4],  'RTE':[3,1,2],  'SNLI':[-1,7,8],  'SST-2':[1,0],  'WNLI':[3,1,2]}
#STS-B skipped for now

if not os.path.isdir('data'):
    os.mkdir('data')
glueDir = 'data/glue/'
if not os.path.isdir(glueDir):
    os.mkdir(glueDir)

if not os.path.isdir('GLUE-baselines'):
    os.system('git clone https://github.com/nyu-mll/GLUE-baselines.git')
    os.system('cp scripts/0.download_glue_data.py GLUE-baselines')
if not os.path.isdir('GLUE-baselines/glue_data'):
    os.system('cd GLUE-baselines && git clone https://github.com/wasiahmad/paraphrase_identification.git && python2 download_glue_data.py --data_dir glue_data --tasks all --path_to_mrpc=paraphrase_identification/dataset/msr-paraphrase-corpus')

for task in idxs:
    for split in ['train', 'dev']:
        readFirst = False
        if task == 'CoLA':
            readFirst = True
        readFile = 'GLUE-baselines/glue_data/'+task+'/'+split +'.tsv'
        if task == 'MNLI' and split == 'dev':
            readFile = 'GLUE-baselines/glue_data/MNLI/'+split +'_matched.tsv'
        elif task == 'MNLI-mis':
            if split == 'dev':
                readFile = 'GLUE-baselines/glue_data/MNLI/'+split +'_mismatched.tsv'
            else:
                readFile = 'GLUE-baselines/glue_data/MNLI/'+split +'.tsv'

        if not os.path.isfile(readFile):
            print("error, " + readFile + " not found")
            continue
        
        outFile = open(glueDir + task + '.' + split, 'w')

        for lineIdx, line in enumerate(open(readFile)):
            if not readFirst:# or lineIdx > 20000:
                readFirst = True
                continue
            tok = line.strip().split('\t')
            goldIdx = idxs[task][0]
            if task == 'QQP' and len(tok) != 6:
                continue
            
            if len(idxs[task]) == 2:
                outFile.write(tok[idxs[task][1]] + '\t' + tok[goldIdx] + '\n')
            else:
                outFile.write(tok[idxs[task][1]] + '\t' + tok[idxs[task][2]] + '\t' + tok[goldIdx] + '\n')
                
            #print(tok[goldIdx])
        outFile.close()
        

