### probdistr task-type
[back to main README](../README.md)

The probdistr task-type can be used to predict a distribution over labels. They are expected to sum
to 1.0. A Kullback-Leibler divergence is used, and the logits are returned and used as prediction 
directly. To use the probdistr task-type, you have to define multiple `column_idxs` in a list, like:

```
    "sentiment": {
        "task_type": "probdistr",
        "column_idxs": [1,2]
    }
```
For a file that could look like:
```
This is awesome!  0.95   0.05
```

Currently, for picking the best model f1 score is used as default (over an argmax), and accuracy is also supported.


