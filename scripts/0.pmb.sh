mkdir -p data

git clone https://github.com/RikVN/DRS_parsing.git
cp DRS_parsing/parsing/layer_data/gold/en/train.conll data/pmb.train
cp DRS_parsing/parsing/layer_data/gold/en/dev.conll data/pmb.dev
cp DRS_parsing/parsing/layer_data/gold/en/test.conll data/pmb.test
