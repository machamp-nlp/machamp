### Results UD
[back to main README](../README.md)

We provide results on UD version 2.7 and 2.8 (as of now, if you need another version, e-mail the first author). These results are obtained by running the scripts in `scripts/2.*`. This means that they are trained without multi-words, but evaluated with multi-words for fair comparison!Furthermore, they are trained with the default settings, and the average over 3 seeds.

The results for version 2.8 are shared on (just replace the 8 with a 7 for the older scores):

* http://www.itu.dk/people/robv/data/machamp/preds-2.8.tar.gz
* https://github.com/machamp-nlp/machamp/results/

The first link contains the full output as well as the output of the [official conllu evaluation script](http://universaldependencies.org/conll18/conll18_ud_eval.py). The latter link contains one file per UD-version and task, and contains the scores of the 5 models used in the paper for each dataset averaged over 3 seeds.


