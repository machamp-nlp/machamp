wget https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.en
wget https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/train.vi

wget https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2012.en
wget https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2012.vi

wget https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2013.en
wget https://nlp.stanford.edu/projects/nmt/data/iwslt15.en-vi/tst2013.vi

paste train.en train.vi > data/iwslt15.envi.train
paste tst2012.en tst2012.vi > data/iwslt15.envi.dev
paste tst2013.en tst2013.vi > data/iwslt15.envi.test

rm train.en train.vi tst2012.en tst2012.vi tst2013.en tst2013.vi
