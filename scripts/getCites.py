import re
import os
import myutils

def findEnd(data, idx):
    nesting = 1
    for i in range(idx+1, len(data)):
        for char in data[i]:
            if char == '}':
                nesting = nesting - 1
            if char == '{':
                nesting = nesting + 1
            if nesting == 0:
                return i + 1
    return len(data) 

def getYear(data, beg, idx):
    for i in range(beg, end):
        if 'year' in data[i]:
            result = re.search('[0-9][0-9][0-9][0-9]', data[i])
            return int(data[i][result.start():result.end()]) # why not just .string?
    return 0

counter = 0

for base in ['data/ud-treebanks-v' + myutils.UDversion + '/', 'data/ud-treebanks-v2.extras/']:
    for udDir in os.listdir(base):
        readmePath = base + udDir + '/README.md'
        if not os.path.isfile(readmePath):
            readmePath = base + udDir + '/README.txt'
        if not os.path.isfile(readmePath):
            continue
        data = open(readmePath).readlines()
        founds = {}
        for lineIdx, line in enumerate(data):
            if re.search("@[a-zA-Z]*{", line) != None:
                end = findEnd(data, lineIdx)
                year = getYear(data, lineIdx, end)
                founds[year] = data[lineIdx:end]
        if len(founds) > 0:
            counter += 1
            mostRecentCite = sorted(founds)[-1]
            outFile = open(base + udDir + '/cite.bib', 'w')
            outFile.write(''.join(founds[mostRecentCite]))
            outFile.close()
print(counter)
print("Remember to fix UD_Turkish-BOUN")
