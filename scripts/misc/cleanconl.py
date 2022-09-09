import os
import sys

if not os.path.isfile('scripts/ud-conversion-tools/conllu_to_conll.py'):
    os.system('cd scripts && git clone https://github.com/bplank/ud-conversion-tools.git && cd ../')


def rm_multiwords(path):
    new_data = []
    for line in open(path):
        line = line.strip('\n')
        if line == '' or line[0] == '#':
            new_data.append(line)
        else:
            tok = line.split('\t')
            tok[8] = '_'
            new_data.append('\t'.join(tok))
    outFile = open(path, 'w')
    for line in new_data:
        outFile.write(line + '\n')
    outFile.close()


def clean_file(conll_file):
    print('cleaning ' + conll_file)
    os.system(
        'python3 scripts/ud-conversion-tools/conllu_to_conll.py ' + conll_file + ' TMP --replace_subtokens_with_fused_forms --print_comments --remove_deprel_suffixes --output_format conllu')
    os.system('mv TMP ' + conll_file)


for path in sys.argv[1:]:
    rm_multiwords(path)
    clean_file(path)

# no words:
# ['UD_Arabic-NYUAD', 'UD_English-ESL', 'UD_French-FTB', 'UD_Hindi_English-HIENCS', 'UD_Mbya_Guarani-Dooley', 'UD_Japanese-BCCWJ', 'UD_Norwegian-NynorskLIA']
