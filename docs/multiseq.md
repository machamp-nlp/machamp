### multiseq task-type
[back to main README](../README.md)

For some tasks it is not known in advance how many labels can be expected per 
instance. In particular this task-type is focusing on sequence labeling. This 
could be the case for example when having multiple annotators, or when having
a hierarchy, like in nested NER. For example the phrase `Australian open' could
be annotated as LOCderiv (Australian) and MISC (Australian open). Usually the data
annotated with two layers would be represented like:

```
Australian  B-MISC   B-LOCderiv
open        I-MISC   O
```

In MaChAmp, because we accept an arbitrary number of labels, we adapt a slightly
different format:

```
Australian  B-MISC|B-LOCderiv
open        I-MISC
```

As can be seen, multiple labels are joined with a `|' in between. The O 
does not have to be used if there is already another label. This task types 
allows for two additional parameters: 

* `threshold`: The threshold which decides which labels to include; if it is set lower, 
it will be more likely to output multiple labels per instance. 1.0 means that it will
only pick the highest scoring candidate, and with 0.0 it ourputs almost all labels.
* `max_heads`: the maximum number of labels to output

For this task-type the default evaluation metric is `multi_span_f1`, however, this is
word-level accuracy excluding O tags, which is due to the usecase of the original 
implementation. We suggest to not directly report this score, but it should be fine for
model selection.

For more information we refer to the paper: [Biomedical Event Extraction as 
Sequence Labeling](http://robvandergoot.com/doc/beesl.pdf)


