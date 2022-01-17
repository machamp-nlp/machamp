### Loss weights
[back to main README](../README.md)

In mult-task setups, a `loss_weight` can be used to give priority over some (or one) of the tasks.
This could be for example because it is less important, easier, or because it has more training data.
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


#### Class weights

Another parameter that could be especially useful when the label distribution on the task level is strongly unbalanced is `class_weights`. By defining it, you can easily tell the classifier to give more importance to some labels (typically, the under-represented ones) compared to others. This will help in learn better representations for them. If `class_weights` is not defined in your config file for your task, the class weights will not be used or computed. Possible values for `class_weights` are:

- `balanced`: a string; the class weights for each label are calculated automatically by MaChAmp based on the training label distribution, by setting each weight to `weight = num_samples / num_labels*label_count` (**preferred**, as it is the standard way in literature to set class weights),
- `{"class1": weight1, ..., "classN": weightN}`: a dictionary `str->float` to set your custom class weights "offline". All the labels need a float value;

A configuration example for a task with name `MYTASK-LABELS` and balanced class weights is presented in the following:

```
"tasks": {
    "MYTASK-LABELS": {
        "task_type": "classification",
        // ...
        "class_weights": "balanced"
    }
}
```

Currently, we fully support the most common use case for class weights, i.e., the `classification` task type, whereas the `mlm`, `seq2seq` and raw text task types are excluded due to the large label space they handle.
