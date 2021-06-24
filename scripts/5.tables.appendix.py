import os
from allennlp.common import Params
import myutils
import statistics


names = {'UD_English-AAE-UD-v1':'en_aae', 'UD_English-ConvBank': 'en_convbank', 'UD_English-MoNoise':'en_monoise', 'UD_English-Tweebank2':'en_tweebank2', 'UD_French-ExtremeUGC0.6.2':'fr_extremeugc', 'UD_Frisian_Dutch-FAME':'qfn_fame', 'UD_German-tweeDE':'de_tweede', 'UD_Singlish-SingPar':'en_singpar'}
for udPath in ['data/ud-treebanks-v' + myutils.UDversion + '/']:
    for UDdir in os.listdir(udPath):
        if not UDdir.startswith("UD"): 
            continue
        train, dev, test = myutils.getTrainDevTest(udPath + UDdir)
        names[UDdir] = test.split('/')[-1].split('-')[0]
        
PROXIES = {}
for line in open('scripts/proxies.txt'):
    tok = line.strip().split()
    PROXIES[tok[0]] = tok[1]

def getSize(path):
    counter = 0
    if path == '':
        return counter
    for line in open(path):
        if len(line) < 3 or line[0] == '#':
            continue
        else:
            counter += 1
    return counter

UDSIZE = {}
for udPath in ['data/ud-treebanks-v' + myutils.UDversion + '/', 'data/ud-treebanks-v2.extras/']:
    for UDdir in os.listdir(udPath):
        if not UDdir.startswith("UD"): 
            continue
        train, dev, test = myutils.getTrainDevTest(udPath + UDdir)
        size = getSize(train)
        UDSIZE[UDdir] = size

largeCites = {'ckt_hse', 'el_gdt', 'lt_hse', 'orv_torot', 'pl_lfg', 'ru_taiga', 'ta_ttb', 'en_monoise', 'qfn_fame'}
allBibs = []
def getCitation(udPath):
    if os.path.isfile(udPath + '/cite.bib'):
        fullBib = ''.join(open(udPath + '/cite.bib').readlines())
        if fullBib not in allBibs:
            allBibs.append(fullBib)
        if names[udPath.split('/')[-1]] in largeCites:
            return '\\resizebox{4cm}{!}{\\cite{' + fullBib.split('\n')[0].split('{')[1].split(',')[0] + '}}'
        else:
            return '{\\small \\cite{' + fullBib.split('\n')[0].split('{')[1].split(',')[0] + '}}'
    else:
        return ''
        
 
settings = ['self', 'concat', 'concat.smoothed', 'sepDec.smoothed', 'datasetEmbeds.smoothed']
#print(' & '.join(['dataset', 'proxy', 'size'] + [setting.replace('smoothed', 'sm') for setting in settings]) + '\\\\')
header = """\\begin{table*}
 \\resizebox{\\textwidth}{!}{
\\begin{tabular}{p{2.2cm} p{4cm} p{1.5cm} r r r r r r r}

\\toprule
 &  &  &  &  &  & \multicolumn{3}{|c|}{+smoothing} \\\\
dataset & citation & proxy & size & self & conc. & conc. & sepDec & dataEmb\\\\
\\midrule"""

footer = """\\bottomrule
\\end{tabular}}
\\end{table*}
"""
print(header)


# zero (0), small (<1,000), medium (<10,000), large (rest)
groups = [1,1000,10000,99999999999]
groupScores = {}
for setting in settings:
    groupScores[setting] = [[],[],[],[]]
table = []
outDir = 'preds/'

counter = 0
for udPath in ['data/ud-treebanks-v' + myutils.UDversion +'/', 'data/ud-treebanks-v2.extras/']:
    udDirs = {}
    for udDir in sorted(os.listdir(udPath)):
        if not udDir.startswith('UD'):
            continue
        train,dev,test = myutils.getTrainDevTest(udPath + udDir)
        if not myutils.hasColumn(test, 1):
            continue
        udDirs[names[udDir]] = udDir
    for udDir in sorted(udDirs):
        udDir = udDirs[udDir]
        counter += 1
        datasetScores = []
        datasetStdevs = []
        for setting in settings:
            instance_scores = []
            for seed in myutils.seeds:
                if setting == 'self':
                    output = outDir + setting + '.' + udDir + '.test.' + str(seed) + '.conllu.eval'
                else:
                    output = outDir + 'fullUD' + setting + '.' + str(seed) + '.' + udDir + '.test.' + str(seed) + '.conllu.eval'
    
                if os.path.isfile(output):
                    score = float(open(output).readline().strip().split()[-1])
                else:
                    print("NF", output)
                    score = 0.0
                instance_scores.append(score)
            score = sum(instance_scores)/len(instance_scores)
            stddev = statistics.pvariance(instance_scores)
            for i in range(len(groups)):
                if UDSIZE[udDir] < groups[i]:
                    groupScores[setting][i].append(score)
                    break
            datasetScores.append(score)
            datasetStdevs.append(stddev)
        proxy = '---'
        if udDir in PROXIES:
            proxy = names[PROXIES[udDir]].replace('_', '\\_')

        #scoresStr = ['{:.1f}$\pm${:.1f}'.format(score,stddev) for score, stddev in zip(datasetScores, datasetStdevs)]
        maxIdx = datasetScores.index(max(datasetScores))
        scoresStr = ['{:.1f}'.format(score) for score, stddev in zip(datasetScores, datasetStdevs)]
        scoresStr[maxIdx] = '\\textbf{' + scoresStr[maxIdx] + '}'
        print(' & '.join([names[udDir].replace('_','\\_'), getCitation(udPath + udDir) , proxy, f"{UDSIZE[udDir]:,}"] + scoresStr) + ' \\\\')

        if counter %50 == 0:
            print(footer)
            print(header)
    if udPath == 'data/ud-treebanks-v' + myutils.UDversion +'/':
        print('\\midrule')

print('\n'.join(footer.split('\n')[:-2]))
print('\\caption{LAS scores from official conll2018 script on test splits of all UD datasets we could obtain, averaged over 3 random seeds. Size refers to number of sentences in the training split. Results for single dataset trained models, and our 4 multi-task strategies. The last 12 rows contain datasets that are either available without words on the official Universal Dependencies website or are not officialy submitted.}')
print('\\label{tab:allUD}')
print('\n'.join(footer.split('\n')[-2:]))
outFile = open('papers.bib', 'w')
for item in allBibs:
    outFile.write(item + '\n')
outFile.close()

print()
print("""Model\\textbackslash Size & 0 & $<$1k & $<$10k & $>$10k   \\
\midrule""")

printNames = ['Single', 'All', 'Smoothed', 'Dataset embed.$^*$', 'Sep. decoder$^*$']
for setting, printName in zip(settings, printNames):
    print(' & '.join([printName] + ['{:.1f}'.format(sum(sizeScores)/len(sizeScores)) for sizeScores in groupScores[setting]]) + ' \\\\')

print()
glue = Params.from_file('configs/glue.json')
for idx, setting in enumerate(['glue', 'glue.smoothSampling', 'glue.single']):
    scores = []
    for task in glue:
        if task == 'WNLI':
            continue
        size = len(open(glue[task]['train_data_path']).readlines())
        output = outDir + setting + '.' + task + '.eval'
        if os.path.isfile(output):
            score = float(open(output).readline().split()[0])
        else:
            score = 0.0
        scores.append((size, task, score))
    scores.sort()
    if idx == 0:
        print(' & '.join([''] + [item[1] for item in scores]) + ' \\\\')
        print(' & '.join([''] + [str(item[0]) for item in scores]) + ' \\\\')
    
    print(' & '.join([setting] + ['{:.1f}'.format(item[2] * 100) for item in scores]) + ' \\\\')
    #print(setting, round(sum(100 * scores)/len(scores), 2))

