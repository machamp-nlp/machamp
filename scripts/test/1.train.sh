python3 train.py --name test.ewt --dataset_configs configs/ewt.json
python3 train.py --name test.qnli --dataset_configs configs/qnli.json
python3 train.py --name test.ner --dataset_configs configs/ner.json
python3 train.py --name test.nlu --dataset_configs configs/nlu.json
python3 train.py --name test.multiseq --dataset_configs configs/multiseq.json
python3 train.py --name test.sts --dataset_configs configs/sts.json
python3 train.py --name test.mlm --dataset_configs configs/mlm.json 
python3 train.py --name test.multitask --dataset_configs configs/ewt.json configs/qnli.json
python3 train.py --name test.multitask-sequential --sequential --dataset_configs configs/ewt.json configs/qnli.json
python3 train.py --name test.multitask-div0 --dataset_configs configs/ewt.json configs/qnli.json --parameters_config scripts/test/params-div-0.json
python3 train.py --name test.multitask-div1 --dataset_configs configs/ewt.json configs/qnli.json --parameters_config scripts/test/params-div-1.json
python3 train.py --name test.multitask-nodiv0 --dataset_configs configs/ewt.json configs/qnli.json --parameters_config scripts/test/params-nodiv-0.json
python3 train.py --name test.multitask-nodiv1 --dataset_configs configs/ewt.json configs/qnli.json --parameters_config scripts/test/params-nodiv-1.json
