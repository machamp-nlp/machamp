python3 aae.py TwitterAAE-UD-v1/wh250_gold.conllu > dev2.conllu
python3 aae.py TwitterAAE-UD-v1/aa250_gold.conllu > test2.conllu
udapy -s ud.FixPunct < dev2.conllu > dev3.conllu
udapy -s ud.FixPunct < test2.conllu > test3.conllu

mv dev3.conllu dev.conllu
mv test3.conllu test.conllu

