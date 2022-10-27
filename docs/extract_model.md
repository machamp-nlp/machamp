### Extracting a model after finetuning
[back to main README](../README.md)

If you would like to re-use your MaChAmp finetuned model in another toolkit, you can extract the 
`transformers` model. We provide an example script in `scripts/misc/extract_automodel.py`, which
needs to be ran from the root of this repo. Usage is as follows:

```
cp scripts/misc/extract_automodel.py .
python3 extract_automodel.py logs/ewt/*/model.pt mBERT_finetuned_on_EWT
```

Now the models including its configuration and tokenizer will be written in a folder titled 
`mBERT_finetuned_on_EWT`.


