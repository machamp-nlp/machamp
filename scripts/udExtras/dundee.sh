cp ../corpora/parsing/ud/ud-treebanks-v2.extras/UD_English-Dundee/dundee_treebank.conllu ../corpora/parsing/ud/ud-treebanks-v2.extras/UD_English-Dundee/dundee_treebank.backup
python3 scripts/udExtras/fixLabels.py ../corpora/parsing/ud/ud-treebanks-v2.extras/UD_English-Dundee/dundee_treebank.conllu > out1
python3 scripts/udExtras/fixRoot.py out1 > out2
python3 scripts/udExtras/fixunicycles.py out2 > out3
udapy -s ud.FixPunct < out3 > ../corpora/parsing/ud/ud-treebanks-v2.extras/UD_English-Dundee/dundee_treebank-test.conllu
