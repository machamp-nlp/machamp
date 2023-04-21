### Change evaluation metrics

[back to main README](../README.md)

MaChAmp supports a variety of metrics for task types. Specifically:

* **accuracy** (`acc`): for the task types [seq](seq.md), [seq_bio](seq_bio.md), and [classification](classification.md)
  ;
* **micro F1 score** (`f1_micro`): for the task types [seq](seq.md), [seq_bio](seq_bio.md),
  and [classification](classification.md);
* **macro F1 score** (`f1_macro`): for the task types [seq](seq.md), [seq_bio](seq_bio.md),
  and [classification](classification.md);
* **binary F1 score** (`f1_binary`): for the task types [seq](seq.md), [seq_bio](seq_bio.md),
  and [classification](classification.md);
* **span-based F1 score** (`span_f1`): for the task type [seq_bio](seq_bio.md) and [seq](seq.md);
* **labeled attachment score** (`las`): for the task type [dependency](dependency.md);
* **perplexity** (`perplexity`): for the task type [mlm](mlm.md);
* **average distance** (`avg_dist`): for the task type [regression](mlm.md)
* **multilabel accurcay** (`multi_acc`): for the task types [multi_seq](multiseq.md) and [multi_clas](multiclas.md)

You can set/check the default metrics used for each task in the parameters configuration file (
default=`configs/params.json`). Alternatively, you can set the `'metric'` keyword per task. To use micro-f1 for POS
tagging for example:

```
{
    "UD": {
        "train_data_path": "data/ewt.train",
        "dev_data_path": "data/ewt.dev",
        "word_idx": 1,
        "tasks": {
            "upos": {
                "task_type": "seq",
                "column_idx": 3,
                "metric": "f1_micro"
            }
        }
    }
}
```

Note: sometimes it is desirable to have multiple metrics logged, for instance if you want to optimize for text classification using macro-f1 but also know the micro-f1 and accuracy scores. To do so, just add a (per-task) `additional_metrics` key with either a list of metric names (list of strings) or just a metric name (string):

```
{
    "UD": {
        "train_data_path": "data/ewt.train",
        "dev_data_path": "data/ewt.dev",
        "word_idx": 1,
        "tasks": {
            "upos": {
                "task_type": "seq",
                "column_idx": 3,
                "metric": "f1_micro",
                "additional_metrics": ["f1_micro", "accuracy"] // or "additional_metrics": "f1_micro"
            }
        }
    }
}
```

