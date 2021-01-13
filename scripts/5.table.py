import os
from allennlp.common import Params
import myutils

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

udPath = 'data/ud-treebanks-v2.7/'
UDSIZE = {}
for UDdir in os.listdir(udPath):
    train, dev, test = myutils.getTrainDevTest(udPath + UDdir)
    if 'Iceland' in UDdir or 'UD_Old_Russian-RNC' in UDdir or not myutils.hasColumn(test, 1):
        continue
    size = getSize(train)
    UDSIZE[UDdir] = size


# zero (0), small (<1,000), medium (<10,000), large (rest)
groups = [1,1000,10000,99999999999]
singleScores = {}
overviewTable = []
outDir = 'preds/'
settings = ['self', 'concat', 'concat.smoothed', 'sepDec.smoothed', 'datasetEmbeds.smoothed']
for setting in settings:
    scores = [[],[],[],[]]
    for UDdir in UDSIZE:
        output = outDir + ('' if setting == 'self' else 'fullUD') + setting + '.' + UDdir + '.dev.conllu.eval'
        if os.path.isfile(output):
            score = float(open(output).readline().strip().split()[-1])
        else:
            output = outDir + ('' if setting == 'self' else 'fullUD') + setting + '.' + UDdir + '.test.conllu.eval'
            if os.path.isfile(output):
                score = float(open(output).readline().strip().split()[-1])
            else:    
                score = 0.0
                print("NF", output)
        for i in range(len(groups)):
            if UDSIZE[UDdir] < groups[i]:
                scores[i].append(score)
                break
        #scores.append(score)
        if UDdir not in singleScores:
            singleScores[UDdir] = []
        singleScores[UDdir].append(score)
        
    overviewTable.append(' & '.join([setting] + ['{:.1f}'.format(sum(setScores)/max(len(setScores),1)) for setScores in scores]) + ' \\\\')

print(' & '.join(['', 'size'] + settings) + '\\\\')
for UDdir in sorted(singleScores):
    scores = ['\\textbf{' + '{:.2f}'.format(score) + '}' if score == max(singleScores[UDdir]) else '{:.2f}'.format(score) for score in singleScores[UDdir]]
    size = '{:,}'.format(0 if UDdir not in UDSIZE else UDSIZE[UDdir])
    print(' & '.join([UDdir.replace('_', '\\_'), size] + scores) + '\\\\')

print()
print(' & '.join(['0', '<1,000', '<10,000', '>10,000']) + ' \\\\')
print('\n'.join(overviewTable))
print()
glue = Params.from_file('configs/glue.json')
for idx, setting in enumerate(['glue', 'glue.smoothSampling', 'glue.single']):
    scores = []
    for task in glue:
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


