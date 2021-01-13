LG=$1

mkdir -p data/nmt
cd data/nmt

for SPLIT in train dev test
do
    wget http://data.statmt.org/opus-100-corpus/v1.0/supervised/en-$LG/opus.en-$LG-$SPLIT.$LG
    wget http://data.statmt.org/opus-100-corpus/v1.0/supervised/en-$LG/opus.en-$LG-$SPLIT.en
    paste opus.en-$LG-$SPLIT.en opus.en-$LG-$SPLIT.$LG > opus.en-$LG-$SPLIT
    rm opus.en-$LG-$SPLIT.en opus.en-$LG-$SPLIT.$LG
done

cd ../../

