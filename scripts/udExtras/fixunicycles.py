import sys

for lineIdx, line in enumerate(open(sys.argv[1])):
    tok = line.strip().split('\t')
    if len(tok) == 10 and tok[6] == tok[0]:
        tok[6] == str(int(tok[6])+1)
        print('\t'.join(tok))
    elif lineIdx +1 == 6067 and tok[1] =='to':
        tok[6] == str(int(tok[6])-1)
        print('\t'.join(tok))
    elif lineIdx +1 == 6068 and tok[1] =='talk':
        tok[6] == str(int(3))
        print('\t'.join(tok))
    else:
        print(line.strip())

