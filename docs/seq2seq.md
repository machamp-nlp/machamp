### seq2seq task-type
[back to main README](../README.md)

For seq2seq task type, we use an autoregressive RNN decoder *(Transformers decoder is a WIP)* with encoder-decoder dot-product attention,
the datasets should be tab sepearated in such format:

```
I think the weather is great	أعتقد أن الطقس رائع اليوم
This movie is bad	هذا الفيلم سيئ
```

Then you can create a dataset configuration for translation on the above dataset (English-Arabic):

```
{
    "NMT": {
        "train_data_path": "data/en-ar.train",
        "validation_data_path": "data/en-ar.dev",
        "sent_idxs": [0],
        "tasks": 
        {
            "en-ar": 
            {
                "task_type": "seq2seq",
                "column_idx": 1
            }
        }
    }
}
```

And you can translate the other way around (Arabic-English) without needing to change the dataset itself, only by swapping columns idxs:

```
{
    "NMT": {
        "train_data_path": "data/en-ar.train",
        "validation_data_path": "data/en-ar.dev",
        "sent_idxs": [1],
        "tasks": 
        {
            "ar-en": 
            {
                "task_type": "seq2seq",
                "column_idx": 0
            }
        }
    }
}
```


where `sent_idx` is the column index for source sentences and `column_idx` is the column index for the target sentences.


For editing seq2seq sepcific model parameters such as beam size and number of decoder layers, you can modify the model [configuration file](../configs/params.json) in `model -> decoders -> seq2seq`.

```
     "seq2seq": {
		"type": "machamp_seq2seq_decoder",
        "attention": "dot_product",
        "beam_size": 6,
        "max_decoding_steps": 128,
        "target_decoder_layers": 2,
        "target_embedding_dim": 512,
        "use_bleu": true
     }
	 
```

For tokenizing target sentences, we currently support two options, the first one is to use the same pre-trained encoder tokenizer (e.g. mBERT wordpieces) on target sentences (`dataset_reader` in [configs](../configs/params.json)), and the second one is basic whitespace and punctuation tokenization ([configs](../configs/params.tgt-words.json)). 
