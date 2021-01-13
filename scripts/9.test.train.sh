python3 train.py --name test.ewt --dataset_config configs/ewt.json --device 0
python3 train.py --name test.qnli --dataset_config configs/qnli.json --device 0
python3 train.py --name test.ner --dataset_config configs/ner.json --device 0
python3 train.py --name test.nlu --dataset_config configs/nlu.json --device 0
python3 train.py --name test.multiseq --dataset_config configs/multiseq.json --device 0
python3 train.py --name test.mlm --dataset_config configs/mlm.json --device 0
python3 train.py --name test.nmt --dataset_config configs/nmt.json --device 0
python3 train.py --name test.all --dataset_config configs/all.json --parameters_config configs/params.smoothSampling.json --device 0
