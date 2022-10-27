### Fine-tuning on a MaChAmp model

[back to main README](../README.md)

As opposed to earlier versions of MaChAmp (and Udify), we now define the layer attention per task. 
This allows for more flexibility, and lets the model choose which parts of the encoder to focus on for
each task (optimal weights are probably different for different types of tasks). Accordingly, the `layer_to_use`
parameter is now defined in the dataset configuration. It should be noted that the input word embeddings are also
a layer; so to use perform attention over all layers in a 12-layer BERT model, the dataset configuration would
look like:

```
"UD_English-EWT": {
    "train_data_path": "../data/ud-treebanks-v2.5.singleToken/UD_English-EWT/en_ewt-ud-train.conllu",
    "dev_data_path": "../data/ud-treebanks-v2.5.singleToken/UD_English-EWT/en_ewt-ud-dev.conllu",
    "word_idx": 1,
    "tasks": {
        "upos": {
            "task_type": "seq",
            "column_idx": 3,
            "layers_to_use": [0,1,2,3,4,5,6,7,8,9,10,11,12]
            }
    }
}
```

Note that the default `layers_to_use` use is set to `[-1]`, and it can be changed in the [hyperparameters](hyper.md) (decoders.default_decoder.layers_to_use)
