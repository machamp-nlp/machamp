# TODO why are some in their own folder?

if [ ! -f data/wiki/fy.train ]; then
    ./scripts/misc/wiki.sh fy
fi

if [ ! -f data/ewt.train ]; then
    ./scripts/0.prep.ewt.sh
fi

if [ ! -f data/glue/QNLI.train ]; then
    python3 scripts/0.prep.glue.py
fi

if [ ! -f data/iwslt15.envi.dev ]; then
    ./scripts/0.prep.nmt.iwslt15.sh
fi

if [ ! -f data/nlu/en/train-en.conllu ]; then
    ./scripts/0.prep.nlu.sh
fi

if [ ! -f data/pmb.train ]; then
    ./scripts/0.prep.pmb.sh
fi

if [ ! -f data/de_news_dev.tsv ]; then
    wget https://github.com/bplank/DaNplus/raw/master/data/de_news_train.tsv -O data/de_news_train.tsv
    wget https://github.com/bplank/DaNplus/raw/master/data/de_news_dev.tsv -O data/de_news_dev.tsv
fi

