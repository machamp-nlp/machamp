sed -i "s;^Corrected by Sarah$;;g" data/ud-treebanks-v2.7.extras*/UD_French-ExtremeUGC0.6.2/LoL.In-game.tout.parsed.gold.morfetted.gold_tagged.tobeUDed.tobecorrected_Corrected_by_Sarah_FR__V0.6.1.2.test.conllu

manually fixed line 59, 117, 204, 429, 801
grep -v "^#" data/ud-treebanks-v2.7.extras/UD_French-ExtremeUGC0.6.2/LoL.In-game.tout.parsed.gold.morfetted.gold_tagged.tobeUDed.tobecorrected_Corrected_by_Sarah_FR__V0.6.1.2.test.conllu > tmp
mv tmp data/ud-treebanks-v2.7.extras/UD_French-ExtremeUGC0.6.2/LoL.In-game.tout.parsed.gold.morfetted.gold_tagged.tobeUDed.tobecorrected_Corrected_by_Sarah_FR__V0.6.1.2.test.conllu 

Manually fixed some other irregularities (multiple tabs/not enough columns, multiple roots)
