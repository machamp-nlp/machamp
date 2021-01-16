### Change evaluation metrics
[back to main README](../README.md)

MaChAmp supports a variety of metrics for task types. Specifically:

* **accuracy** (`acc`): for the task types [seq](seq.md), [seq_bio](seq_bio.md), and [classification](classification.md);
* **micro F1 score** (`micro-f1`): for the task types [seq](seq.md), [seq_bio](seq_bio.md), and [classification](classification.md);
* **macro F1 score** (`micro-f1`): for the task types [seq](seq.md), [seq_bio](seq_bio.md), and [classification](classification.md);
* **span-based F1 score** (`span_f1`): for the task type [seq_bio](seq_bio.md);
* **multi span-based F1 score** (`multi_span_f1`): for the task type [multiseq](multiseq.md);
* **labeled attachment score** (`las`): for the task type [dependency](dependency.md);
* **perplexity** (`perplexity`): for the task type [mlm](mlm.md);
* **bilingual evaluation understudy** (`bleu`): for the task type [seq2seq](seq2seq).