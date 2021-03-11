import sys

tree = []

for line in open(sys.argv[1]):
    tok = line.strip().split('\t')
    if line.strip() == '':
        if len(tree) != 0:
            firstRoot = ''
            for wordIdx in range(0, len(tree)):
                if tree[wordIdx][7] == 'root':
                    if tree[wordIdx][6] != '0':
                        tree[wordIdx][7] = 'parataxis'
                    elif firstRoot == '':
                        firstRoot = tree[wordIdx][0]
                    else:
                        tree[wordIdx][7] = 'parataxis'
                        tree[wordIdx][6] = firstRoot
                
            for sent in tree:
                print('\t'.join(sent))
            print()

            tree = []
    elif line.startswith('#'):
        print(line, end='')
    else:
        tree.append(tok)

