### Change embeddings
[back to main README](../README.md)

Thanks to huggingface and the Allennlp 1.0 update, we can now easily swap the embeddings.
The steps for this are as follows:

1. Find the huggingface name of the embeddings you want to use: [https://huggingface.co/models](https://huggingface.co/models)
2. Change the transformer model name in `configs/params.json` correspondingly (or make your [own hyperparameters file](hyper.md). Also update the dimensions if necessary (these can be found in the config files on the huggingface embeddings page).
3. That's it!, now train your model as normal

