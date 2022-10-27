### seq task-type

[back to main README](../README.md)

The `seq` task-type is used for standard sequence labeling tasks. It assumes one
label per word. It can for example be used to do POS tagging:

```
I   PRON
choose    VERB
you PRON
Pickachu  PROPN
!   PUNCT

Use       VERB
thunderbolt   NOUN
!   PUNCT
```

The dataset configuration file should looks like this (assuming the data is stored in `machamp.train`):

```
{
    "POS": {
        "train_data_path": "machamp.train",
        "dev_data_path": "machoke.dev",
        "word_idx": 0,
        "tasks": {
            "upos": {
                "task_type": "seq",
                "column_idx": 1
            }
        }
    }
}
```


