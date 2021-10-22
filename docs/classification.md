### classification task-type
[back to main README](../README.md)

For the classification task, the system reads one instance per line. Similarly
as for word-level tasks, you have to define a `sent_idxs` for each dataset, and
a `column_idx` for each task. If for example you have a file with `labels <TAB>
sentence`, `sent_idxs` should be `1` and `column_idx=0`. It should be noted
that you can use a list for the `sent_idxs`, if you want to use multiple input
utterances (they will be concatenated internally, with a SEP token inbetween).

```
{
    "RTE": {
        "train_data_path": "data/glue/RTE.train",
        "validation_data_path": "data/glue/RTE.dev",
        "sent_idxs": [0,1],
        "tasks": {
            "rte": {
                "column_idx": 2,
                "task_type": "classification"
            }
        }
    }
}

```

### Joint sequence labeling and classification within dataset

We also support classification of labels inside a conll-like format. In this
situation, the name of the task should match the label used in the comment, and
the `column_idx` should be set to -1.

If your data looks like this:
```
# text: tell me the weather report for half moon bay
# intent: weather/find
# slots: 12:26:weather/noun,31:44:location
1	tell	weather/find	NoLabel
2	me	weather/find	NoLabel
3	the	weather/find	NoLabel
4	weather	weather/find	B-weather/noun
5	report	weather/find	I-weather/noun
6	for	weather/find	NoLabel
7	half	weather/find	B-location
8	moon	weather/find	I-location
9	bay	weather/find	I-location

```

The dataset configuration should be (note that `intent' should match with line 2 of the data file):

```
{
    "NLU": {
        "train_data_path": "data/nlu/en/train-en.conllu",
        "validation_data_path": "data/nlu/en/eval-en.conllu",
        "word_idx": 1,
        "tasks": {
            "slots": {
                "task_type": "seq",
                "column_idx": 3
            },
            "intent": {
                "task_type": "classification",
                "column_idx": -1
            }
        }
    }
}
```
