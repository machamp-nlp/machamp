import myutils
import os
import math
import matplotlib.pyplot as plt

plt.style.use('scripts/rob.mplstyle')

def getNumSents(path):
    counter = 0
    for line in open(path):
        if len(line) < 2:
            counter += 1
    return counter

datasets = {}
for udPath in ['data/ud-treebanks-v2.7/', 'data/ud-treebanks-v2.7.extras/']:
    for udDir in os.listdir(udPath):
        if not udDir.startswith('UD'):
            continue
        train,dev,test = myutils.getTrainDevTest(udPath + udDir)
        if train == '':
            continue
        if not myutils.hasColumn(test, 1):
            continue
        sents = getNumSents(train)
        datasets[udDir] = sents


def getSmoothedSizes(smooth_factor, datasets):
    # calculate new size based on smoothing
    sizes = [datasets[x] for x in datasets]
    new_sizes = []
    total_size = sum(sizes)
    total_new_prob = 0.0
    for dataset in datasets:
        size = datasets[dataset]
        pi = size/total_size
        total_new_prob += math.pow(pi, smooth_factor)
    
    for dataset in datasets:
        size = datasets[dataset]
        pi = size/total_size
        prob = (1/pi) * (math.pow(pi, smooth_factor)/total_new_prob)
        new_sizes.append(int(size * prob))
    return list(reversed(sorted(new_sizes)))



fig, ax = plt.subplots(figsize=(8,5), dpi=300)
sizes = []
names = []
for dataset in sorted(datasets, key=datasets.get, reverse=True):
    sizes.append(datasets[dataset])
    names.append(dataset)
    

ax.bar(range(len(sizes)), sizes, label=r'$\alpha$=1.0')

for smooth in [0.0, 0.25,.50,.75]:
    ax.plot(range(len(sizes)), getSmoothedSizes(smooth, datasets), label=r'$\alpha$=' + str(smooth))

ax.set_ylim((0,50000))
ax.set_xlim((-1,len(sizes)-1))
ax.set_xticks(())
ax.set_ylabel('Size')
ax.set_xlabel('Treebanks')
leg = ax.legend()
leg.get_frame().set_linewidth(1.5)

fig.savefig('smoothing.pdf', bbox_inches='tight')



