import sys
import os

outFile = open(sys.argv[1] + '.fixed', 'w')
idx = 1 
for line in open(sys.argv[1]):
    if len(line) > 2:
        tok = line.strip().split('\t')
        tok[0] = str(idx)
        idx += 1
        tok.extend(['_', '_'])
        outFile.write('\t'.join(tok) + '\n')
    else:
        outFile.write('\n')
        idx = 1


