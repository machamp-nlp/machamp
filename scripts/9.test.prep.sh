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

if [ ! -f data/nmt/opus.en-fy-train ]; then
    ./scripts/0.prep.nmt.sh fy
fi

if [ ! -f data/nlu/en/train-en.conllu ]; then
    ./scripts/0.prep.nlu.sh
fi

if [ ! -f data/pmb.train ]; then
    ./scripts/0.prep.pmb.sh
fi

if [ ! -f data/NER-de-train.tsv ]; then
    wget "https://drive.google.com/u/0/uc?id=1Jjhbal535VVz2ap4v4r_rN1UEHTdLK5P&export=download" -o data/NER-de-train.tsv
    wget "https://drive.google.com/u/0/uc?id=1ZfRcQThdtAR5PPRjIDtrVP7BtXSCUBbm&export=download" -o data/NER-de-dev.tsv
fi

