mkdir wmt14
cd wmt14/

wget http://www.statmt.org/wmt13/training-parallel-europarl-v7.tgz
wget http://statmt.org/wmt14/training-parallel-nc-v9.tgz
wget http://www.statmt.org/wmt13/training-parallel-commoncrawl.tgz
wget http://www.statmt.org/wmt14/dev.tgz
wget http://www.statmt.org/wmt14/test-filtered.tgz
tar -zxvf training-parallel-europarl-v7.tgz
tar -zxvf training-parallel-nc-v9.tgz
tar -zxvf training-parallel-commoncrawl.tgz
tar -zxvf dev.tgz
tar -zxvf test-filtered.tgz

mv commoncrawl* training/

#rm training-parallel-europarl-v7.tgz
#rm training-parallel-nc-v9.tgz
#rm training-parallel-commoncrawl.tgz
#rm dev.tgz
#rm test-full.tgz


paste training/news-commentary-v9.de-en.en training/news-commentary-v9.de-en.de > training/news.ende
paste training/europarl-v7.de-en.en training/europarl-v7.de-en.de > training/europarl.ende
paste training/commoncrawl.de-en.en training/commoncrawl.de-en.de > training/commoncrawl.ende
cat training/news.ende training/europarl.ende training/commoncrawl.ende > wmt14.ende.train

paste dev/newstest2013.en dev/newstest2013.de > wmt14.ende.dev

cat test/newstest2014-deen-src.en.sgm  | cut -d ">" -f 2 | cut -d "<" -f 1 | grep -v "^$" > test.en
cat test/newstest2014-deen-ref.de.sgm  | cut -d ">" -f 2 | cut -d "<" -f 1 | grep -v "^$" > test.de
paste test.en test.de > wmt14.ende.test

cd ../
