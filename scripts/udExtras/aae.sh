udapy -s ud.FixPunct < wh250_gold.conllu > dev.conllu
udapy -s ud.FixPunct < aa250_gold.conllu > test.conllu

python3 fixLabels.py TwitterAAE-UD-v1/dev.conllu > fixed
mv fixed TwitterAAE-UD-v1/dev.conllu

python3 fixLabels.py TwitterAAE-UD-v1/test.conllu > fixed
mv fixed TwitterAAE-UD-v1/test.conllu

