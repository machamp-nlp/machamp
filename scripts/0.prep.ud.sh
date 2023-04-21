mkdir -p data
wget https://lindat.mff.cuni.cz/repository/xmlui/bitstream/handle/11234/1-4923/ud-treebanks-v2.11.tgz
tar -zxvf ud-treebanks-v2.11.tgz
mv ud-treebanks-v2.11 data
cp -r data/ud-treebanks-v2.11 data/ud-treebanks-v2.11.singleToken
python3 scripts/misc/cleanconl.py data/ud-treebanks-v2.11.singleToken/*/*conllu

