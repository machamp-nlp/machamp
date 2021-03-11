python3 fixunicycles.py Sing_Par/TALLIP19_dataset/treebank/gold_pos/train.ext.conll > Sing_Par/train.conll
python3 fixunicycles.py Sing_Par/ACL17_dataset/treebank/gold_pos/dev.conll > Sing_Par/dev.conll
python3 fixunicycles.py Sing_Par/ACL17_dataset/treebank/gold_pos/test.conll > Sing_Par/test.conll


python3 fixLabels.py Sing_Par/test.conll > fixed
mv fixed Sing_Par/test.conll
python3 fixLabels.py Sing_Par/dev.conll > fixed
mv fixed Sing_Par/dev.conll
python3 fixLabels.py Sing_Par/train.conll > fixed
mv fixed Sing_Par/train.conll
