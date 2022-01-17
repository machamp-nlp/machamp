### Label balancing
[back to main README](../README.md)

This parameter is especially useful when the label distribution on the task level is strongly unbalanced is `class_weights`. By defining it, you can easily tell the classifier to give more importance to some labels (typically, the under-represented ones) compared to others. This will help in learn better representations for them. If `class_weights` is not defined in your config file for your task, the class weights will not be used or computed. Possible values for `class_weights` are:

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
