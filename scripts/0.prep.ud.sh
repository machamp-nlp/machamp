wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-3424/ud-treebanks-v2.7.tgz
tar -zxvf ud-treebanks-v2.7.tgz
mv ud-treebanks-v2.7 data
python3 scripts/misc/cleanconl.py data/ud-treebanks-v2.7/*/*conllu

