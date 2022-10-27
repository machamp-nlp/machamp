mkdir -p data
wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-4758/ud-treebanks-v2.10.tgz
tar -zxvf ud-treebanks-v2.10.tgz
mv ud-treebanks-v2.10 data
cp -r data/ud-treebanks-v2.10 data/ud-treebanks-v2.10.singleToken
python3 scripts/misc/cleanconl.py data/ud-treebanks-v2.10.singleToken/*/*conllu

