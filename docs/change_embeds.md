### Change embeddings

[back to main README](../README.md)

The steps for changing the language model are as follows:

1. Find the huggingface name of the embeddings you want to
   use: [https://huggingface.co/models](https://huggingface.co/models). If you want to use a local model, you can just
   use the path to the embeddings as name
2. Change the transformer model name (`transformer_model`) in `configs/params.json` correspondingly (or make
   your [own hyperparameters file](hyper.md).
3. That's it!, now train your model as normal

