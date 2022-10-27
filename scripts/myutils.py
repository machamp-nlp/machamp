import os
import _jsonnet
import json

UDversion = '2.9'
seeds = ['1']
def getTrainDevTest(path):
    train = ''
    dev = ''
    test = ''
    for conlFile in os.listdir(path):
        if conlFile.endswith('conllu'):
            if 'train' in conlFile:
                train = path + '/' + conlFile
            if 'dev' in conlFile:
                dev = path + '/' + conlFile
            if 'test' in conlFile:
                test = path + '/' + conlFile
    return train, dev, test

def hasColumn(path, idx, threshold=.1):
    total = 0
    noWord = 0
    for line in open(path).readlines()[:5000]:
        if line[0] == '#' or len(line) < 2:
            continue
        tok = line.strip().split('\t')
        if tok[idx] == '_':
            noWord += 1
        total += 1
    return noWord/total < threshold

def getModel(name):
    modelDir = 'logs/'
    nameDir = modelDir + name + '/'
    if os.path.isdir(nameDir):
        for modelDir in reversed(os.listdir(nameDir)):
            modelPath = nameDir + modelDir + '/model.pt'
            if os.path.isfile(modelPath):
                return modelPath
    return ''

def load_json(path: str):
    """
    Loads a jsonnet file through the json package and returns a dict.
    
    Parameters
    ----------
    path: str
        the path to the json(net) file to load
    """
    return json.loads(_jsonnet.evaluate_snippet("", '\n'.join(open(path).readlines())))

multiRegressive = ['Helsinki-NLP/opus-mt-mul-en', 'bigscience/bloom-560m', 'facebook/mbart-large-50', 'facebook/mbart-large-50-many-to-many-mmt', 'facebook/mbart-large-50-many-to-one-mmt', 'facebook/mbart-large-50-one-to-many-mmt', 'facebook/mbart-large-cc25', 'facebook/mgenre-wiki', 'facebook/nllb-200-distilled-600M', 'facebook/xglm-564M', 'facebook/xglm-564M', 'google/byt5-base', 'google/byt5-small', 'google/canine-c', 'google/canine-s', 'google/mt5-base', 'google/mt5-small', 'sberbank-ai/mGPT']
multiAutoencoder = ['Peltarion/xlm-roberta-longformer-base-4096', 'bert-base-multilingual-cased', 'bert-base-multilingual-uncased', 'cardiffnlp/twitter-xlm-roberta-base', 'distilbert-base-multilingual-cased', 'google/rembert', 'microsoft/infoxlm-base', 'microsoft/infoxlm-large', 'microsoft/mdeberta-v3-base', 'setu4993/LaBSE', 'studio-ousia/mluke-base', 'studio-ousia/mluke-base-lite', 'studio-ousia/mluke-large', 'studio-ousia/mluke-large-lite', 'xlm-mlm-100-1280', 'xlm-roberta-base', 'xlm-roberta-large']
too_large = ['facebook/xlm-roberta-xxl', 'facebook/xlm-roberta-xl', 'google/byt5-xxl', 'google/mt5-xxl', 'google/mt5-xl', 'google/byt5-xl', 'google/byt5-large', 'google/mt5-large', 'facebook/nllb-200-1.3B', 'facebook/nllb-200-3.3B', 'facebook/nllb-200-distilled-1.3B']


