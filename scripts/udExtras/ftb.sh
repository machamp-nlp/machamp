cp FTBUDCONLLU+DEEPFEAT/* .

sed -i "s;	void=y	;	_	;g" fr_ftb-ud-dev.conllu.withwords+deepfeat
sed -i "s;|void=y	;	;g" fr_ftb-ud-dev.conllu.withwords+deepfeat
sed -i "s;|void=y|;|;g" fr_ftb-ud-dev.conllu.withwords+deepfeat
sed -i "s;void=y|;;g" fr_ftb-ud-dev.conllu.withwords+deepfeat

sed -i "s;	void=y	;	_	;g" fr_ftb-ud-test.conllu.withwords+deepfeat
sed -i "s;|void=y	;	;g" fr_ftb-ud-test.conllu.withwords+deepfeat
sed -i "s;|void=y|;|;g" fr_ftb-ud-test.conllu.withwords+deepfeat
sed -i "s;void=y|;;g" fr_ftb-ud-test.conllu.withwords+deepfeat

sed -i "s;	void=y	;	_	;g" fr_ftb-ud-train.conllu.withwords+deepfeat
sed -i "s;|void=y	;	;g" fr_ftb-ud-train.conllu.withwords+deepfeat
sed -i "s;|void=y|;|;g" fr_ftb-ud-train.conllu.withwords+deepfeat
sed -i "s;void=y|;;g" fr_ftb-ud-train.conllu.withwords+deepfeat


sed -i "s;	diat=passif	;_;g" fr_ftb-ud-dev.conllu.withwords+deepfeat
sed -i "s;|diat=passif	;;g" fr_ftb-ud-dev.conllu.withwords+deepfeat
sed -i "s;|diat=passif|;|;g" fr_ftb-ud-dev.conllu.withwords+deepfeat
sed -i "s;diat=passif|;;g" fr_ftb-ud-dev.conllu.withwords+deepfeat

sed -i "s;	diat=passif	;_;g" fr_ftb-ud-test.conllu.withwords+deepfeat
sed -i "s;|diat=passif	;;g" fr_ftb-ud-test.conllu.withwords+deepfeat
sed -i "s;|diat=passif|;|;g" fr_ftb-ud-test.conllu.withwords+deepfeat
sed -i "s;diat=passif|;;g" fr_ftb-ud-test.conllu.withwords+deepfeat

sed -i "s;	diat=passif	;_;g" fr_ftb-ud-train.conllu.withwords+deepfeat
sed -i "s;|diat=passif	;;g" fr_ftb-ud-train.conllu.withwords+deepfeat
sed -i "s;|diat=passif|;|;g" fr_ftb-ud-train.conllu.withwords+deepfeat
sed -i "s;diat=passif|;;g" fr_ftb-ud-train.conllu.withwords+deepfeat

python3 Tweebank/fixRoot.py UD_French-FTB/fr_ftb-ud-train.conllu.withwords+deepfeat > fixed.conllu && mv fixed.conllu UD_French-FTB/fr_ftb-ud-train.conllu.withwords+deepfeat



