### Predict on raw data
[back to main README](../README.md)

Prediction on raw data is very similar as [predicting on full datasets](predict_data.md), the only change is that you add `--raw_text` as a paramenter. The expected input format whitespace separated word, with a newline to separate the sentences:

```
I choose you Machamp !
Machamp, use mega punch !
```

Assuming this text is saved in `battle.txt`, the command would be:

```
python3 predict.py logs/ewt/<DATE>/model.tar.gz battle.txt battle.pred.conllu --raw_text
```

Note that this is not extensively tested for all task combinations, and for now is its also safer to use it with --dataset. 

