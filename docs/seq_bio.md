### seq_bio task-type
[back to main README](../README.md)

The usage of this task type is similar as [seq](seq.md). However, this task type assumes that
your data is in BIO format, and also guarantees that the output will be in valid BIO format.
Under the hood it is using a masked CRF, which disallows ill-formed output. Performance can be 
expected to be higher for span labeling tasks (like NER) compared to the standard `seq` decoder.

