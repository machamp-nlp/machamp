mkdir -p data
wget https://github.com/UniversalDependencies/UD_English-EWT/archive/r2.9.tar.gz
tar -zxvf r2.9.tar.gz
cp UD_English-EWT-r2.9/en_ewt-ud-train.conllu data/ewt.train
cp UD_English-EWT-r2.9/en_ewt-ud-dev.conllu data/ewt.dev
cp UD_English-EWT-r2.9/en_ewt-ud-test.conllu data/ewt.test
python3 scripts/misc/cleanconl.py data/ewt.train data/ewt.dev data/ewt.test
rm -rf r2.9.tar.gz UD_English-EWT-r2.9

