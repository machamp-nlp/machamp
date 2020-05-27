import sys

def cleanFile(conlFile):
    print('cleaning ' + conlFile)
    lines = open(conlFile).readlines()
    newLines = []
    for line in lines:
        tok = line.strip().split('\t')
        if line[0] != '#' and ('-' in tok[0] or '.' in tok[0]):
            continue
        else:
            newLines.append(line)
    outFile = open(conlFile, 'w')
    for line in newLines:
        outFile.write(line)
    outFile.close()

for path in sys.argv[1:]:
    cleanFile(path)

    

