### Adding a new task-type
[back to main README](../README.md)

Adding a new task type can involve some effort, so I suggest to get in touch with robv@itu.dk.

Basically it involves three steps:

* Check if an existing dataset reader can match your task type; if it does, make sure it uses
the correct one: `read_function` in `_read()`. If there is no reader that matches your data
structure, you have write a new one.
* Add the task-type to the `configs/param.json`, including any parameters you need to use for 
the following step
* The main task is to include a task decoder. If you are lucky, you can draw inspiration from
[the official AllenNLP models](https://github.com/allenai/allennlp-models). If you use any of
these, the main adaptation is to adapt the `forward()` function; instead of encoding the text
there, you should use an `embedded_text` variable. The adaptation of `__init__` should be 
straightforward, and `get_metrics` can be copied from another machamp decoder. I would suggest 
to take a look at one of the existing decoders in `machamp/models/` and compare it to an 
Allennlp version. The `tag_decoder` and `sentence_decoder` are based on models from the 
[AllenNLP repo](https://github.com/allenai/allennlp/tree/master/allennlp/models).
* Make sure the `MachampModel` model forward pass passes the right encoding (sentence/word-level)
to the right decoder.


