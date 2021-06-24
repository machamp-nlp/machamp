### Models
[back to main README](../README.md)

We provide our models trained on the whole UD 2.7 with the default seed:

* [concat](http://itu.dk/people/robv/data/machamp/machamp-ud-concat.tar.gz): Trained on a concatenation of all available UD 2.7 training datasets, with default MaChAmp hyperparameters.
* [smoothed](http://itu.dk/people/robv/data/machamp/machamp-ud-concat-smoothed.tar.gz): Same as above, with multinomial smoothing (0.5) enabled)
* [dataset embeddings (smoothed)](http://itu.dk/people/robv/data/machamp/machamp-ud-datasetEmbeds-smoothed.tar.gz): Also added dataset embeddings
* [separate decoders (smoothed)](http://itu.dk/people/robv/data/machamp/machamp-ud-sepDec-smoothed.tar.gz): Uses a separate decoder for each dataset

These models are trained on 125 treebanks and 74 languages.
After these are downloaded and extracted, we refer to the main [README](../README.md) for information on how to use them for prediction. (note that for the latter two models, one has to define which dataset to use with `--dataset `).

### Even larger model
We also provide a model where the training data consists of the train+test split. **This data should not be used when handling UD data**, but it can be useful for projects where one simply needs a universal parser for other data:

* [smoothed](http://itu.dk/people/robv/data/machamp/machamp-ud-gigantamax.tar.gz)

This parser is trained on 106 languages (195 treebanks), and is based on xlm-roberta-large.
