python3 fixRoot.py en-ud-tweet-dev.conllu > en-ud-tweet-dev.fixed.conllu
python3 fixRoot.py en-ud-tweet-test.conllu > en-ud-tweet-test.fixed.conllu
python3 fixRoot.py en-ud-tweet-train.conllu > en-ud-tweet-train.fixed.conllu

mkdir orig
mv en-ud-tweet-test.conllu en-ud-tweet-dev.conllu en-ud-tweet-train.conllu orig/

