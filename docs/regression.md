### regression task-type

[back to main README](../README.md)

The regression task predict a (floating point) number. It uses a mean square error loss, and expects
annotation on the utterance level. An example annotated file looks like:

```
Greece votes in crucial election	Greece looks set for repeat election	1.8
```

Than the dataset configuration would like like:

```
{
    "STS-B": {
        "train_data_path": "data/GLUE-baselines/glue_data/STS-B/train.tsv",
        "dev_data_path": "data/GLUE-baselines/glue_data/STS-B/dev.tsv",
        "sent_idxs": [0,1],
        "tasks": {
            "similarity": {
                "column_idx": 2,
                "task_type": "regression"
            }
        }
    }
}
```

Currently, we only have the average distance (`avg_dist`) for the regression task.

