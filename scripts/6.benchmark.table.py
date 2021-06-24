import os
from allennlp.common import Params
import myutils
import statistics

largeCites = {'ckt_hse', 'el_gdt', 'lt_hse', 'orv_torot', 'pl_lfg', 'ru_taiga', 'ta_ttb', 'en_monoise', 'qfn_fame'}
allBibs = []
def getCitation(udPath):
    if os.path.isfile(udPath + '/cite.bib'):
        fullBib = ''.join(open(udPath + '/cite.bib').readlines())
        if fullBib not in allBibs:
            allBibs.append(fullBib)
        if names[udPath.split('/')[-1]] in largeCites:
            return '\\resizebox{4cm}{!}{\\cite{' + fullBib.split('\n')[0].split('{')[1].split(',')[0] + '}}'
        else:
            return '{\\small \\cite{' + fullBib.split('\n')[0].split('{')[1].split(',')[0] + '}}'
    else:
        return ''
        
header = """\\begin{table*}
 \\resizebox{\\textwidth}{!}{
\\begin{tabular}{p{2.2cm} p{4cm} p{1.5cm} r r r r r r r}

\\toprule
 &  &  &  &  &  & \multicolumn{3}{|c|}{+smoothing} \\\\
dataset & citation & proxy & size & self & conc. & conc. & sepDec & dataEmb\\\\
\\midrule"""
print(header)

data = open('results.' + myutils.UDversion + '.csv').readlines()

def highestBold(row):
    floats = [float(x) for x in row[4:]]
    for i in range(len(floats)):
        if floats[i] == max(floats):
            row[4+i] = '\\textbf{' + row[4+i] + '}'
    row[3] = "{:,}".format(int(float(row[3])))
    return row

allScores = [[],[],[],[],[]]
for row in data[1:]:
    tok = row.strip().split(',')
    del tok[0]
    if tok[3] == '---':
        allScores[0].append(0.0)
    else:
        allScores[0].append(float(tok[3]))
    allScores[1].append(float(tok[4]))
    allScores[2].append(float(tok[5]))
    allScores[3].append(float(tok[6]))
    allScores[4].append(float(tok[7]))
    print(' & '.join(highestBold(tok)) + ' \\\\')

tok = ['Avg.', '', ''] + ['{:.2f}'.format(sum(x)/len(x)) for x in allScores]
print(' & '.join(highestBold(tok)) + ' \\\\')
 

footer = """\\bottomrule
\\end{tabular}}
\\end{table*}
"""
print(footer)
