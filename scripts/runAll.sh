python3 scripts/0.prep.glue.py
./scripts/0.prep.ud.sh
./scripts/0.prep.nmt.wmt14.sh
./scripts/0.prep.nmt.iwslt15.sh

python3 scripts/1.tune.py > 1.tune.sh
chmod +x 1.tune.sh
./1.tune.sh

python3 scripts/2.ud.train.py > 2.train.sh
chmod +x 2.train.sh
./2.train.sh

python3 scripts/2.ud.pred.py > 2.pred.sh
chmod +x 2.pred.sh
./2.pred.sh

python3 scripts/2.ud.eval.py

python3 scripts/3.glue.train.py > 3.train.sh
chmod +x 3.train.sh
./3.train.sh

python3 scripts/3.glue.pred.py > 3.pred.sh
chmod +x 3.pred.sh
./3.pred.sh

python3 scripts/3.glue.eval.py

