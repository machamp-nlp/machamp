"""
The main UDify predictor to output conllu files
"""

import copy

from typing import List
from overrides import overrides

from allennlp.common.util import JsonDict, sanitize
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from machamp.dataset_readers.lemma_edit import apply_lemma_rule

@Predictor.register("machamp_predictor")
class MachampPredictor(Predictor):
    """
    Predictor for a UDify model that takes in a sentence and returns
    a single set conllu annotations for it.
    """
    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self.model = model

    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    @overrides
    def predict_batch_instance(self, instances: List[Instance]) -> List[JsonDict]:
        if "@@UNKNOWN@@" not in self._model.vocab._token_to_index["lemmas"]: #TODO Why lemmas?
            # Handle cases where the labels are present in the test set but not training set
            for instance in instances:
                self._predict_unknown(instance)
        outputs = self._model.forward_on_instances(instances)
        return sanitize(outputs)

    @overrides
    def predict_instance(self, instance: Instance) -> JsonDict:
        if "@@UNKNOWN@@" not in self._model.vocab._token_to_index["lemmas"]: #TODO why lemmas?
            # Handle cases where the labels are present in the test set but not training set
            self._predict_unknown(instance)
        outputs = self._model.forward_on_instance(instance)
        return sanitize(outputs)

    def _predict_unknown(self, instance: Instance):
        """
        Maps each unknown label in each namespace to a default token
        :param instance: the instance containing a list of labels for each namespace
        """
        def replace_tokens(instance: Instance, namespace: str, token: str):
            if namespace not in instance.fields:
                return

            instance.fields[namespace].labels = [label
                                                 if label in self._model.vocab._token_to_index[namespace]
                                                 else token
                                                 for label in instance.fields[namespace].labels]

        #TODO how to generalize this?
        replace_tokens(instance, "lemmas", "↓0;d¦")
        replace_tokens(instance, "feats", "_")
        replace_tokens(instance, "xpos", "_")
        replace_tokens(instance, "upos", "NOUN")
        replace_tokens(instance, "head_tags", "case")

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        """
        Expects JSON that looks like ``{"sentence": "..."}``.
        Runs the underlying model, and adds the ``"words"`` to the output.
        """
        sentence = json_dict["sentence"]
        tokens = sentence.split()
        return self._dataset_reader.text_to_instance(tokens)

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        #import pprint
        #pprint.pprint(outputs)
        #exit(1)
        allTasks = list(outputs['colIdxs'])
        firstTask = allTasks[0] if allTasks[0] != 'wordIdx' else allTasks[1]
        firstTaskIdx = outputs['tasks'].index(firstTask)
        if outputs['task_types'][firstTaskIdx] == 'classification':
            tok = outputs['fullData']
            #import pprint
            #pprint.pprint(outputs)
            #print(tok)
            for taskIdx, task in enumerate(outputs['tasks'] if type(outputs['tasks']) is list else [outputs['tasks']]):
                if task in outputs['colIdxs']:
                    colIdx = outputs['colIdxs'][task]
                    tok[colIdx] = outputs[task][0]
            #print(tok)
            #print()
            return '\t'.join(tok) + '\n'

        else: # sequence labeling
            output_lines = []
            # Add comments
            numComments = len(outputs['fullData']) - len(outputs['words'])
            for commentIdx in range(numComments):
                output_lines.append('\t'.join(outputs['fullData'][commentIdx]))
            # For each word
            for wordIdx in range(len(outputs['words'])):               
                # collect all information in `tok'
                size = max(outputs['colIdxs'].values()) + 1
                if type(outputs['fullData'][numComments + wordIdx]) == list:
                    tok = ['_'] * max(size, len(outputs['fullData'][numComments + wordIdx]))
                else:
                    tok = ['_'] * size

                if outputs['copy_other_columns']:
                    tok = outputs['fullData'][numComments + wordIdx]
                else:
                    # add original word
                    tok[outputs['colIdxs']['wordIdx']] = outputs['words'][wordIdx]
                for task, taskType in zip(outputs['tasks'], outputs['task_types']):
                    if task not in outputs['colIdxs']:
                        continue #TODO, is this the best solution?
                    taskIdx = outputs['colIdxs'][task]
                    if taskType == 'string2string':
                        tok[taskIdx] = apply_lemma_rule(outputs['words'][wordIdx], outputs[task][wordIdx])
                        #if tok[taskIdx] == '':#TODO, is this a good default?
                        #    tok[taskIdx] = '_'
                    elif taskType == 'dependency':
                        tok[taskIdx] = str(outputs['predicted_heads'][wordIdx])
                        if len(tok) -1 == taskIdx:
                            tok.append('')
                        tok[taskIdx+1] = outputs['predicted_dependencies'][wordIdx]
                    elif taskType == 'multiseq':
                        tok[taskIdx] = '$'.join(outputs[task][wordIdx])
                    elif taskType == 'seq':
                        tok[taskIdx] = outputs[task][wordIdx]
                    else:
                        log.warning("Error, taskType " + taskType + ' is not defined in predictor')
                        exit(1)
                output_lines.append('\t'.join(tok))
                
        return '\n'.join(output_lines)  + '\n\n'


