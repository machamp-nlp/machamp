### Fine-tuning on a MaChAmp model

[back to main README](../README.md)

By default, MaChAmp trains on all datasets jointly. However, for some tasks it
might be beneficial to train sequentially. This means that you first finetune
embeddings for one task, save the parameters, and then re-train on another
task. Only for the first dataset, the original parameters of the embeddings
defined in params.json is loaded, after this they are automatically replaced
with the ones from the previous dataset.

MaChAmp supports two interfaces to this functionality. With the first, we
directly define all the dataset configurations in the order that they should
be used:

```
python3 train.py --dataset_configs configs/en_mlm.json configs/ewt.json --sequential
```

In the second method, we train each model separately. This also allows to
use a separate params.json for each finetuning step (in which replacing the
embeddings has no effect, since they are replaced). This is done as follows:

```
python3 train.py --dataset_configs configs/en_mlm
python3 train.py --dataset_configs configs/ewt.json --finetune logs/en_mlm/2020.12.08_21.04.10/
```

