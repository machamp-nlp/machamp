if [ ! -f data/wiki/fy.train ]; then
    ./scripts/misc/wiki.sh fy
fi

if [ ! -f data/ewt.train ]; then
    ./scripts/test/0.prep.ewt.sh
fi

if [ ! -f data/GLUE-baselines/glue_data/QNLI/train.tsv ]; then
    python3 scripts/0.prep.glue.py
fi

if [ ! -f data/nmt.wmt14.ende.train ]; then
    ./scripts/test/0.prep.nmt.wmt14.sh
fi

if [ ! -f data/xSID-0.3/en.train.conll ]; then
    ./scripts/test/0.prep.nlu.sh
fi

if [ ! -f data/NER-de-train.tsv ]; then
    ./scripts/test/0.prep.ner.sh
fi

