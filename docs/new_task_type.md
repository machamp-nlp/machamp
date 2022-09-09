u### Adding a new task-type
[back to main README](../README.md)

Adding a new task type can involve some effort, so I suggest to get in touch with robv@itu.dk.

Basically it involves three steps:

* Check if an existing dataset reader can match your task type; if it does, make sure it uses
  the correct one: `read_function` in `machamp/data/machamp_dataset.py`. If there is no reader that matches your data
  structure, you have write a new one.
* Add the task-type to the `configs/param.json`, including any parameters you need to use for
  the following step
* The main task is to include a task decoder. If you are lucky, you can draw inspiration from
  [the AllenNLP models](https://github.com/allenai/allennlp-models). I would suggest
  to take a look at the most similar of the existing decoders in `machamp/models/`.
* Make sure the `MachampModel` model forward pass passes the right encoding (sentence/word-level)
  to the right decoder.


