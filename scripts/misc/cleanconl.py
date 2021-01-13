import sys
import os

if not os.path.isfile('scripts/ud-conversion-tools/conllu_to_conll.py'):
    os.system('cd scripts && git clone https://github.com/bplank/ud-conversion-tools.git && cd ../')

def rmEUD(path):
    newData = []
    for line in open(path):
        line = line.strip('\n')
        if line == '' or line[0] == '#':
            newData.append(line)
        else:
            tok = line.split('\t')
            tok[8] = '_'
            newData.append('\t'.join(tok))
    outFile = open(path, 'w')
    for line in newData:
        outFile.write(line + '\n') 
    outFile.close()

def cleanFile(conlFile):
    print('cleaning ' + conlFile)
    os.system('python3 scripts/ud-conversion-tools/conllu_to_conll.py ' + conlFile + ' TMP --replace_subtokens_with_fused_forms --print_comments --output_format conllu')
    os.system('mv TMP ' + conlFile)

for path in sys.argv[1:]:
    rmEUD(path)
    cleanFile(path)

# no words:
# ['UD_Arabic-NYUAD', 'UD_English-ESL', 'UD_French-FTB', 'UD_Hindi_English-HIENCS', 'UD_Mbya_Guarani-Dooley', 'UD_Japanese-BCCWJ', 'UD_Norwegian-NynorskLIA']

