### Task-specific parameters
[back to main README](../README.md)

At the moment, the only important metric to define on the task-level is `loss_weight`.
This can be used in a multi-task setting, when some tasks should get less priority while
updating the parameters (because it is less important, easier, or because it has more training data).
It can be set as follows:

```
{
    "UD-EWT": {
        "train_data_path": "data/ewt.train",
        "validation_data_path": "data/ewt.dev",
        "word_idx": 1,
        "tasks": {
            "upos": {
                "task_type": "string2string",
                "column_idx": 2,
                "loss_weight": .5,
            },
            "dependency": {
                "task_type": "dependency",
                "column_idx": 6
            }
        }
    }
}
```

The default `loss_weight` is defined in `configs/params.json`, and is `1.0`, meaning that in the example above
the `dependency` task has twice as much importance as `upos` for the loss.

