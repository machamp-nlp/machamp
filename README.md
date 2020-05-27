# MaChAmp: Massive Choice, Ample Tasks

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

[![Machamp](machamp.png)]()

> One arm alone can move mountains. 


This code base is a wrapper for the [AllenNLP]() library with a focus on
multi-task learning.  It has support for training on multiple datasets for
multiple types of tasks:

* sequence labeling
* string conversion (lemmatization)
* dependency parsing
* sentence classification

This code is basically a generalization of [Udify](), which is focused on 
parsing and tagging UD data.

### Installation
To install all necessary packages run:

```
pip3 install --user -r requirements.txt
```

## Training
To train the model, you need to write a configuration file. Below we show an
example of such a file for training a model for the English Web Treebank in 
the Universal Dependencies format. 

```
{
    "UD": {
        "train_data_path": "data/ewt.train",
        "validation_data_path": "data/ewt.dev",
        "word_idx": 1,
        "tasks": {
            "lemma": {
                "task_type": "string2string",
                "column_idx": 2
            },
            "upos": {
                "task_type": "seq",
                "column_idx": 3
            },
            "xpos": {
                "task_type": "seq",
                "column_idx": 4
            },
            "morph": {
                "task_type": "seq",
                "column_idx": 5
            },
            "dependency": {
                "task_type": "dependency",
                "column_idx": 6
            }
        }
    }
}

```

Every dataset needs at least a (unique) name, and a `train_data_path`,
`validation_data_path`, and `word_idx`. The `word_idx` tells the model in which
column the input words can be found. 

Every task requires a (unique) name, a `task_type` and a `column_idx`. The `task_type`
should be one of `seq`, `string2string`, `dependency`, `classification`. These are
described in more detail below. Another useful option to add can be `metric`,
which should be one of `acc`, `span_f1`, `macro-f1`, `micro-f1` and `LAS`, we have
already set some logical defaults for the tasks if you do not define this. For
more options, check `default_decoder` in `config/params.json`.

On top of this dataset parameters file, you also need to define the hyperparameters, 
as a starting point you can use `config/params.json`. If you want to use another BERT
model, you can define it in this `params.json`. With both of these config files, 
all that you need to do to train is:

```
python3 train.py --parameters_config configs/params.json --dataset_config configs/ewt.json --device 0 --name ewt
```
You can set `--device -1` to use the cpu. The model will be save in
`logs/ewt/<date>_<time>`. We have prepared several scripts to download data,
and corresponding configuration files, these can be found in `scripts` and the 
`configs` directory.

#### Tasks types
* `seq`: sequence labeling, each instance will be read and predicted as a label for the corresponding word.
* `string2string`: same as sequence labeling, but learns a conversion from the original word to the instance, and uses that as label.
* `dependency`: uses a deep biaffine parser, needs to read from two columns (first head index, second dependency label).
* `classification`: sentence level classification, predicts a label for an utterance of text.

For all the word-level tasks (first five), the data should be formatted similar
to the conllu format; comments start with a `#`, there is one word per line,
and the annotations are directly behind it (tab-separated). Lines are seperated
by an empty line. Make sure all words have the same number of columns. 

For the classification task, the system reads one instance per line. Similarly
as in the previous config example, you have to define a `sent_idxs` for each
dataset (which can correspond to a sentence here), and a `column_idx` for each
task. If for example you have a file with `labels \t sentence`, `sent_idxs`
should be `1` and `column_idx=0`. It should be noted that you can use a list
for the `sent_idxs`, if you want to use multiple input utterances (they will be
concatenated internally, with a SEP token inbetween).


#### Training on multiple datasets
This is rather straightforward, all you have to do is add multiple datasets in
the config file. For example, if we want to do supertagging (from the PMB),
jointly with XPOS tags (from the UD) and RTE (Glue), the config file looks like:

```
{
    "UD": {
        "train_data_path": "ewt.train",
        "validation_data_path": "ewt.dev",
        "word_idx": 1,
        "tasks": {
            "upos": {
                "task_type": "seq",
                "column_idx": 3
            }
        }
    },
    "PMB": {
        "train_data_path": "pmb.train",
        "validation_data_path": "pmb.dev",
        "word_idx": 0,
            "ccg": {
                "task_type": "seq",
                "column_idx": 3
            }
        }
    },
    "RTE": {
        "train_data_path": "data/glue/RTE.train",
        "validation_data_path": "data/glue/RTE.dev",
        "sent_idxs": [0,1],
        "tasks": {
            "rte": {
                "column_idx": 2,
                "task_type": "classification",
                "adaptive": true
            }
        }
    }
}
``` 


## Predicting
For predicting on new data you can use `predict.py`, and provide it with the
model-archive, input data, and an output path:

```
python3 predict.py logs/ewt/<DATE>/model.tar.gz data/twitter/dev.norm predictions/ewt.twitter.out --device 0
```


## Other BERT embeddings
In case you want to use other embeddings, you can follow the following steps:

* Download the embeddings
* Transform them to pytorch format
* Adapt the parameters config file

We will go through these step for using bert-large (English, cased), first run the following commands:

```
cd configs/archive/
wget https://storage.googleapis.com/bert_models/2019_05_30/wwm_cased_L-24_H-1024_A-16.zip
unzip wwm_cased_L-24_H-1024_A-16.zip
mv wwm_cased_L-24_H-1024_A-16 bert-large
pytorch_transformers bert bert-large/bert_model.ckpt bert-large/bert_config.json bert-large/pytorch_model.bin
```

Now change the config filew we suggest to at least adapt the following parameters:
```
dataset_reader["token_indexers"]["bert"]["pretrained_model"] = "configs/archive/bert-large/vocab.txt"
model["token_embedders"]["pretrained_model"] = "configs/archive/bert-large/"
model["encoder"]["input_dim"] = 1024
model["default_decoder"]["encoder"]["input_dim"] = 1024
model["default_decoder"]["layer"] = 24

```

## FAQ
Q: I would like to copy over the annotations of the tasks which are not predicted by machamp. 

A: specify this in the `default_dataset` in params.json, or on the dataset level in the `dataset_config`: `copy_other_columns: true`

Q: This is great!, what should I cite?

A:
```
TODO
```

Q: Amazing stuff!, but I was looking for resources on Machamps language:

A: No problem, we have collected a dataset from utterances transcribed from wild Machamps as well as Machamps belonging to Pokémon trainers. It can be found on TODO

