import os
import sys
from transformers import tokenization_utils

if not os.path.isfile('scripts/ud-conversion-tools/conllu_to_conll.py'):
    os.system('cd scripts && git clone https://github.com/bplank/ud-conversion-tools.git && cd ../')


def rm_multiwords(path):
    new_data = []
    for line in open(path, errors='ignore'):
        line = line.strip('\n')
        if line == '' or line[0] == '#':
            new_data.append(line)
        else:
            tok = line.split('\t')
            if len(tok) == 10:
                tok[8] = '_'
            new_data.append('\t'.join(tok))
    outFile = open(path, 'w')
    for line in new_data:
        outFile.write(line + '\n')
    outFile.close()


def remove_control_chars(in_file, out_file):
    lines = []
    for line in open(in_file):
        lines.append('')
        for char in line:
            if not tokenization_utils._is_control(char):
                lines[-1] += char
    outFile = open(out_file, 'w')
    for line in lines:
        outFile.write(line)
    outFile.close()
    

def clean_file(conll_file):
    print('cleaning ' + conll_file)
    os.system(
        'python3 scripts/ud-conversion-tools/conllu_to_conll.py ' + conll_file + ' TMP --replace_subtokens_with_fused_forms --print_comments --remove_deprel_suffixes --output_format conllu')
    os.system('mv TMP ' + conll_file)
    ##remove_control_chars('TMP', conll_file)


for path in sys.argv[1:]:
    rm_multiwords(path)
    clean_file(path)

# no words:
# ['UD_Arabic-NYUAD', 'UD_English-ESL', 'UD_French-FTB', 'UD_Hindi_English-HIENCS', 'UD_Mbya_Guarani-Dooley', 'UD_Japanese-BCCWJ', 'UD_Norwegian-NynorskLIA']
