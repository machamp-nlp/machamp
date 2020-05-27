import sys
import os
import json
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.style.use('scripts/rob.mplstyle')

if len(sys.argv) < 2:
    print('please provide folder to read results from')
    exit(1)

def getScores(path):
    scores = []
    for i in range(1,81):
        epochPath = path + '/metrics_epoch_' + str(i) + '.json'
        if os.path.isfile(epochPath):
            score = json.load(open(epochPath))['validation_.run/.sum']
            scores.append(score)
        else:
            return scores

fig, ax = plt.subplots(figsize=(8,5), dpi=300)
for path, name in zip(sys.argv[1:], ['MaChAmp',  '+proportional']):
    scores = getScores(path)
    print(path, scores)
    ax.plot(range(1,len(scores)+1), scores, label=name)
ax.set_ylabel('Accuracy (Summed)')
ax.set_xlabel('Epoch')
leg = ax.legend(loc='lower right')
leg.get_frame().set_linewidth(1.5)

plt.savefig('proportional.pdf',bbox_inches='tight')


