### dependency task-type

[back to main README](../README.md)

This is a [deep biaffine parser](https://openreview.net/pdf?id=Hk95PK9le), which is used similarly
as the [seq](seq.md) task type. However, it has one peculiarity, namely that it reads data
from two columns. However, you only have to define the first column (which should be the index of
the head), and then it automatically reads the labels from the column behind it:

```
1	Champ	_	PROPN	_	_	1	vocative	_	_
2	champ	_	VERB	_	_	0	root	_	_
3	Macaaaamp	_	NOUN	_	_	2	obj	_	_
```

```
    "dependency": {
        "task_type": "dependency",
        "column_idx": 6
    }
```

This does not support ellipsis, or word splitting, which are included in the
standard [UD](https://universaldependencies.org/) downloads. Because of this, we include a script
that removes these, and attaches the left-over words to the dependency structure. This can be done
with the scripts `scripts/misc/cleanconl.py`, which as arguments takes a list of conllu files, and
replaces all of these with there cleaned version (**warning**, this replaces the original file)

Furthermore, it should be noted that it does not actually use the word indexes which are present
in standard UD format, but just uses the line index from the file.

