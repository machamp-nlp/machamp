### Dataset Embeddings
[back to main README](../README.md)

Previous work has shown that for neural network models it can be beneficial to encode properties (i.e. origin/language) of the input utterance as an embedding, and concatenate it to the word embedding before passing it throught the model. This was first done by embedding the language ([Ammar et al., 2016](https://www.aclweb.org/anthology/Q16-1031/)), but later also to indicate different data sources ([Stymne et al., 2018](https://www.aclweb.org/anthology/P18-2098/)). We use the universal term dataset embeddings in MaChAmp.

To enable dataset embeddings, they have to be activated in both the hyperparameters and the dataset configuration.
In the dataset configuration, they can be added on the dataset level by specifying its column index:

```
{
    "UD": {
        "train_data_path": "data/ewt.train",
        "validation_data_path": "data/ewt.dev",
        "word_idx": 1,
        "dataset_embed_idx": 9,
        "tasks": {
            "upos": {
                "task_type": "seq",
                "column_idx": 3
            }
        }
    }
}
```

In the parameters configuration the size of the embeddins should be set in `model/dataset_embeds_dim`. The default is 0, which means disabled. Commonly used sizes in previous work are 16 and 32 (this value should probably correlate with the number of datasets).

In the above example, the dataset id will be read from the 9th column. The standard format of the UD misc column is used, so it is first split on the | character, and then looks for the keyword `dataset_embed=`. The part after the `=` will be used as dataset identifier. By setting the index to -1, the name of the dataset in the configuration file is used as identifier (in this case `UD'). This can similarly be done for sentence level tasks.

The data should then look as follows:

```
# newdoc id = weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000
# sent_id = weblog-juancole.com_juancole_20051126063000_ENG_20051126_063000-0001
# text = Al-Zaman : American forces killed Shaikh Abdullah al-Ani, the preacher at the mosque in the town of Qaim, near the Syrian border.
1	Al	Al	PROPN	NNP	Number=Sing	0	root	_	SpaceAfter=No|dataset_embed=blog
2	-	-	PUNCT	HYPH	_	1	punct	_	SpaceAfter=No|dataset_embed=blog
3	Zaman	Zaman	PROPN	NNP	Number=Sing	1	flat	_	dataset_embed=blog
4	:	:	PUNCT	:	_	1	punct	_	dataset_embed=blog
5	American	american	ADJ	JJ	Degree=Pos	6	amod	_	dataset_embed=blog
6	forces	force	NOUN	NNS	Number=Plur	7	nsubj	_	dataset_embed=blog
7	killed	kill	VERB	VBD	Mood=Ind|Tense=Past|VerbForm=Fin	1	parataxis	_	dataset_embed=blog
```

It should be noted that in previous work the dataset embedding was attached to the word embedding before passing it to the encoder. However, because in transformer based embeddings, the embeddings are the encoder, and it is non-trivial to change their input, in MaChAmp the dataset embedding is attached after the encoding.

