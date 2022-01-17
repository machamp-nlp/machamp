### regression task-type
[back to main README](../README.md)

This task-type can be used to predict real (floating point) numbers. The input is expected to be 
one or more sentences. A mean square error is used as loss function, and the only supported metrics
are pearson and spearman correlations. A dataset configuration could look as follows:

```
{
    "STS-B": {
        "train_data_path": "data/GLUE-baselines/glue_data/STS-B/train.tsv",
        "validation_data_path": "data/GLUE-baselines/glue_data/STS-B/dev.tsv",
        "sent_idxs": [
            7,
            9
        ],
        "skip_first_line": true,
        "tasks": {
            "sts-b": {
                "column_idx": 9,
                "task_type": "regression"
            }
        }
    }
}
```

For the STS-B dataset in the GLUE benchmark, which looks like:

```
index	genre	filename	year	old_index	source1	source2	sentence1	sentence2	score
0	main-captions	MSRvid	2012test	0000	none	none	A man with a hard hat is dancing.	A man wearing a hard hat is dancing.	5.000

```



