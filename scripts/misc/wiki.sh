#!/bin/sh
set -e

LG=$1
WIKI_DUMP_NAME=${LG}wiki-latest-pages-articles.xml.bz2
WIKI_DUMP_DOWNLOAD_URL=https://dumps.wikimedia.org/${LG}wiki/latest/$WIKI_DUMP_NAME

# download latest Wikipedia dump in chosen language
echo "Downloading the latest $LG-language Wikipedia dump from $WIKI_DUMP_DOWNLOAD_URL..."
wget -c $WIKI_DUMP_DOWNLOAD_URL
echo "Succesfully downloaded the latest $LG-language Wikipedia dump to $WIKI_DUMP_NAME"

WIKI_DUMP_FILE_IN=$WIKI_DUMP_NAME
WIKI_DUMP_FILE_OUT=${WIKI_DUMP_FILE_IN%%.*}.txt

# clone the WikiExtractor repository
if [ ! -d wikiextractor ]; then
    git clone https://github.com/attardi/wikiextractor.git
    cd wikiextractor
    git reset --hard 16186e290d9eb0eb3a3784c6c0635a9ed7e855c3
    cd ..
fi

# extract and clean the chosen Wikipedia dump
echo "Extracting and cleaning $WIKI_DUMP_FILE_IN to $WIKI_DUMP_FILE_OUT..."
python3 wikiextractor/WikiExtractor.py $WIKI_DUMP_FILE_IN --processes 8 -q -o - \
| sed "/^\s*\$/d" \
| grep -v "^<doc id=" \
| grep -v "</doc>\$" \
> $WIKI_DUMP_FILE_OUT
echo "Succesfully extracted and cleaned $WIKI_DUMP_FILE_IN to $WIKI_DUMP_FILE_OUT"

rm $WIKI_DUMP_FILE_IN

if [ ! -d data/wiki ]; then
    mkdir -p data/wiki
fi

head -n -10000 $WIKI_DUMP_FILE_OUT > data/wiki/$LG.train
tail -n -10000 $WIKI_DUMP_FILE_OUT | head -5000 > data/wiki/$LG.dev
tail -n -5000 $WIKI_DUMP_FILE_OUT > data/wiki/$LG.test

