### Predict on raw data
[back to main README](../README.md)

Prediction on raw data is very similar as [predicting on full datasets](predict_data.md), the only change is that you add --raw_text as a paramenter. The expected input format whitespace separated word, with a newline to separate the sentences:

```
I choose you Machamp!
Machamp, use mega punch!
```

Assuming this text is saved in `battle.txt`, the command would be:

```
python3 predict.py logs/ewt/<DATE>/model.tar.gz battle.txt battle.pred.conllu --raw_text
```

For `seq2seq` and `classification`, raw_text is not fully implemented yet. TODO

