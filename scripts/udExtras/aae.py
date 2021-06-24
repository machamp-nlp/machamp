import sys

for line in open(sys.argv[1]):
    tok = line.strip().split('\t')
    if len(tok) > 5 and tok[7] == '_':
        tok[3] = 'PUNCT'
        tok[7] = 'punct'
    print('\t'.join(tok))
    
