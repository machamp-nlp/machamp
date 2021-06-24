import os
from allennlp.common import Params
import myutils
import statistics
 
names = {'UD_English-AAE-UD-v1':'en_aae', 'UD_English-ConvBank': 'en_convbank', 'UD_English-MoNoise':'en_monoise', 'UD_English-Tweebank2':'en_tweebank2', 'UD_French-ExtremeUGC0.6.2':'fr_extremeugc', 'UD_German-tweeDE':'de_tweede', 'UD_Singlish-SingPar':'en_singpar', 'UD_English-Dundee': 'en_dundee'}
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

def saveScores(metric):
    settings = ['self', 'concat', 'concat.smoothed', 'sepDec.smoothed', 'datasetEmbeds.smoothed']
    columns = ['dataset', 'size', 'proxy'] + settings
    outDir = 'preds' + myutils.UDversion + '/'
    outFile = open('results/scores.' + metric + '-' + myutils.UDversion + '.csv', 'w')
    outFile.write(','.join(columns) + '\n')
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
            datasetScores = []
            datasetStdevs = []
            for setting in settings:
                instance_scores = []
                for seed in myutils.seeds:
                    if setting == 'self':
                        output = outDir + setting + '.' + udDir + '.test.' + str(seed) + '.conllu.eval'
                    else:
                        output = outDir + 'fullUD' + setting + '.' + str(seed) + '.' + udDir + '.test.' + str(seed) + '.conllu.eval'
        
                    score = 0.0
                    if os.path.isfile(output):
                        for line in open(output):
                            if line.startswith(metric):
                                score = float(line.split('|')[3].strip())
                    else:
                        print("NF", output)
                    instance_scores.append(score)
                score = sum(instance_scores)/len(instance_scores)
                stddev = statistics.pvariance(instance_scores)
                datasetScores.append(score)
                datasetStdevs.append(stddev)
            proxy = '---'
            if udDir in PROXIES:
                proxy = names[PROXIES[udDir]].replace('_', '\\_')
    
            
            row = [udDir, names[udDir], getCitation(udPath + udDir), proxy, str(UDSIZE[udDir])]
            row += ['{:.2f}'.format(score) for score in datasetScores]
            outFile.write(','.join(row) + '\n')
    outFile.close()        

saveScores('UPOS')
saveScores('LAS')
saveScores('UAS')
saveScores('UFeats')
saveScores('Lemmas')
outFile = open('papers-' + myutils.UDversion + '.bib', 'w')
for item in allBibs:
    outFile.write(item + '\n')
outFile.close()

