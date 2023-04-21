### Heterogeneous batches

[back to main README](../README.md)

When diverse is set to false in `batching/diverse` of the
[hyperparameters](hyper.md), each batch contains only data from a single
dataset.  When this is set to true, diverse batching (i.e. heterogeneous
batching) is enabled, and the training batches can contain data from a variety
of datasets.  This has shown to be highly beneficial when training on many
datasets; for example for
[Muppet](https://aclanthology.org/2021.emnlp-main.468.pdf).

It should be noted that this setting respects the choosen batch size and
smoothing. `sort_by_size` is not supported, as it would likely bias batches
towards certain dataset. Also, this is not guaranteed to see every instance in
your training data every epoch, even if dataset sampling is set to 1.0 (when
`diverse=False`, this is actually guaranteed), as it actually still does random
sampling.



