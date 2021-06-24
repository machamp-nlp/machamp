import os
import myutils

UDWORDS = {}
for udPath in ['data/ud-treebanks-v' + myutils.UDversion + '/', 'data/ud-treebanks-v' + myutils.UDversion + '.extras/']:
    for UDdir in os.listdir(udPath):
        if not UDdir.startswith("UD"): 
            continue
        train, dev, test = myutils.getTrainDevTest(udPath + UDdir)
        if not myutils.hasColumn(test, 1):
            #print('NOWORDS', test)
            continue
        words = myutils.getWords(train)
        UDWORDS[udPath + UDdir] = set(words)

PROXIES = {}
for UDdir in UDWORDS:
    train, dev, test = myutils.getTrainDevTest(UDdir)
    if train == '':
        testWords = myutils.getWords(test, 10)
        scores = {}
        for proxy in UDWORDS:
            scores[proxy] = myutils.getOverlap(testWords, UDWORDS[proxy])
        PROXIES[UDdir] = sorted(scores, key=scores.get, reverse=True)[0]

outFile = open('scripts/proxies.txt', 'w')
for udPath in sorted(PROXIES):
    outFile.write(udPath.split('/')[-1] + '\t' + PROXIES[udPath].split('/')[-1] + '\n')
outFile.close()

