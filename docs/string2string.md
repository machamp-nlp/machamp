### string2string task-type
[back to main README](../README.md)

The usage of this task type is similar as [seq](seq.md). However, it is not labeling the
target labels directly. Instead it learns a conversion from the original word to the target label.
This strategy is commonly used to convert lemmatization to a sequence labeling task. 
Internally, the model replace a sequence like this:
```
Got  get
ta   to
catch    catch
em   them
all  all
!  !
```

To a sequence like this:
```
Got  ↓0;d¦--+e+t
ta   ↓0;d¦-+o
catch    ↓0;d¦
em   ↓0;d+t+h¦
all  ↓0;d¦
!    ↓0;d¦
```

The labels can be used to convert the word on the left to its corresponding
lemma and vice-versa.  MaChAmp now performs a standard sequence labeler on
these convertion-labels.  During evaluation/prediction it uses the predicted
convertion-labels to generate the target lemmas.

For the [UD English Web
Treebank](https://github.com/UniversalDependencies/UD_English-EWT) for example,
this reduces the label space of the lemma column from 14,909 to 268.

