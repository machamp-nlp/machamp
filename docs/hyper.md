### Hyperparameters

[back to main README](../README.md)

The default hyperparameters are the result of tuning on a wide variety of tasks, and should thus provide
a good starting point in most situations. However, you can tune for your target task to try and improve
performance. The most likely starting point would be [changing the embeddings](change_embeds.md), and/or the
learning rate (lr). For other options of tuning, see `configs/params.json`.

To change any hyperparameter, you can either change its value in `configs/params.json`, or create your
own hyperparameters file. If you create your own, we suggest you copy `configs/params.json`.

```
cp configs/params.json configs/params-champ.json
```

After you changed the parameters, you can now train using `--parameters_config`:

```
python3 train.py --dataset_configs configs/ewt.json --device 0 --name ewt --parameters_config config/params-champ.json
```

Parameters that will likely have an effect on performance include:

* [embeddings](change_embeds.md) which are used
* `batching/batch_size`: You should probably also change `batching/max_tokens' accordingly. We have this parameter, because it
saves a lot of GPU ram.
* `trainer/optimizer/lr` in combination with:
* `trainer/num_epochs`
* `encoder/dropout`


### Decoder dropout
Decoder dropout is defined on the task-type level. It can be set in
`decoders/[task\_type]`.  Currently, decoder dropout is not supported for
language modeling, because it uses the native language modeling head from
Huggingface.  Thanks to Lguyogiro for the implementation.

