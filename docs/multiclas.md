### multiclass task-type

[back to main README](../README.md)

For some tasks it is not known in advance how many labels can be expected per
instance. In particular this task-type is focusing on classification. We assume
that the labels are given with a | as separator:

```
Smell ya later	goodbye|negative
```

This task type has an important additional parameter: 

* `threshold`: The threshold which decides which labels to include; if it is set lower,
  it will be more likely to output multiple labels per instance. 1.0 means that it will
  only pick very certain candidates, and with 0.0 it ourputs all labels (default=.7)

For this task-type the default evaluation metric is `multi_acc`, which is the number of
utterances that get the exact correct set of labels. Note that this task does not work with
the other metrics.


