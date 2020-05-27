import json
import pprint

for datasetFile in ['ewt', 'glue', 'pmb']:
    data = json.load(open('configs/' + datasetFile + '.json'))
    for dataset in data:
        if len(data[dataset]['tasks']) > 1:
            for task in data[dataset]['tasks']:
                newData = {}
                newData['train_data_path'] = data[dataset]['train_data_path']
                newData['validation_data_path'] = data[dataset]['validation_data_path']
                newData['word_idx'] = data[dataset]['word_idx']
                newData['tasks'] = {task: data[dataset]['tasks'][task]}
                fullData = {dataset: newData}
                path = 'configs/' + datasetFile + '.' + task + '.json'
                with open(path, 'wt') as out:
                    pprint.pprint(fullData, stream=out)
                cmd = 'python3 train.py --parameters_config configs/params.json --dataset_config ' + path + ' --device 0 --name ' + datasetFile + '.' + task
                print(cmd)
        else:
            path = 'configs/' + datasetFile + '.' + list(data[dataset]['tasks'])[0] + '.json'
            fullData = {dataset:data[dataset]}
            with open(path, 'wt') as out:
                pprint.pprint(fullData, stream=out)
            cmd = 'python3 train.py --parameters_config configs/params.json --dataset_config ' + path + ' --device 0 --name ' + datasetFile + '.' + list(data[dataset]['tasks'])[0]
            print(cmd)
    cmd = 'python3 train.py --parameters_config configs/params.json --dataset_config configs/' +datasetFile + '.json --device 0 --name ' + datasetFile
    print(cmd)

