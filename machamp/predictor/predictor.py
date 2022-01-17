import logging

from allennlp.common.util import JsonDict
from allennlp.data import DatasetReader, Instance
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor
from overrides import overrides

from machamp.dataset_readers.lemma_edit import apply_lemma_rule

logger = logging.getLogger(__name__)


@Predictor.register("machamp_predictor")
class MachampPredictor(Predictor):
    """
    Predictor for a MaChAmp model that takes in a sentence and returns
    a single set conllu annotations for it.
    """

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        raise NotImplementedError

    def __init__(self, model: Model, dataset_reader: DatasetReader) -> None:
        super().__init__(model, dataset_reader)
        self.model = model
    
    def to_str_sentlevel(self, outputs):
        tok = outputs['full_data']
        # detect if raw input is used
        if type(tok) == str:
            tok = [tok]
            for task_idx, task in enumerate(outputs['tasks'] if type(outputs['tasks']) is list else [outputs['tasks']]):
                if task in outputs['col_idxs']:
                    if type(outputs['col_idx']) == list:
                        for one_col_idx, one_pred in zip(col_idx, outputs[task]):
                            while one_col_idx >= len(tok):
                                tok.append('_')
                            tok[one_col_idx] = str(one_pred)
                    else:
                        while outputs['col_idxs'][task] >= len(tok):
                            tok.append('_')
                        tok[outputs['col_idxs'][task]] = str(outputs[task])
            return '\t'.join(tok) + '\n'
        else:
            for task_idx, task in enumerate(outputs['tasks'] if type(outputs['tasks']) is list else [outputs['tasks']]):
                if task in outputs['col_idxs']:
                    col_idx = outputs['col_idxs'][task]
                    if type(col_idx) == list:
                        for one_col_idx, one_pred in zip(col_idx, outputs[task]):
                            tok[one_col_idx] = str(one_pred)
                    else:
                        tok[col_idx] = str(outputs[task])
            return '\t'.join(tok) + '\n'

    def to_str_seq2seq(self, outputs):
        task = outputs['tasks'][0]
        orig = outputs['full_data'][0]
        translation = ''
        for word in outputs[task][0]:
            if word.startswith('##'):#TODO, this is BERT-based! if we can get a tokenizer here, we could use convert_tokens_to_string: https://stackoverflow.com/questions/66232938/how-to-untokenize-bert-tokens
                word = word[2:]
                translation += word
            else:
                translation = translation + ' ' + word
        return orig + '\t' + translation.strip() + '\n'

    # word level can also include sentence level tasks
    def to_str_wordlevel(self, outputs):
        # comments:
        output_lines = []
        num_comments = len(outputs['full_data']) - len(outputs['tokens'])
        for comment_idx in range(num_comments):
            output_lines.append('\t'.join(outputs['full_data'][comment_idx]))

        # sentence level tasks (are saved in comments):
        for task in outputs['tasks']:
            if task not in outputs['col_idxs']:
                continue  # when predicting only one set with --dataset
            task_idx = outputs['col_idxs'][task]
            if task_idx == -1:
                task_str = '# ' + task + ': '
                idx = [task_str if x[0].startswith(task_str) else '' for x in outputs['full_data']]
                idx = idx.index(task_str)
                output_lines[idx] = task_str + outputs[task]
        # word level tasks:
        for word_idx in range(len(outputs['tokens'])):               
            # collect all information in `tok'
            if type(outputs['full_data'][num_comments + word_idx]) == list:
                tok = outputs['full_data'][num_comments + word_idx]
            else:
                # TODO get real max size, should be saved during training?
                size = max(outputs['col_idxs'].values()) + 1
                tok = ['_'] * size
                tok[outputs['col_idxs']['word_idx']] = outputs['full_data'][num_comments + word_idx]

            for task, task_type in zip(outputs['tasks'], outputs['task_types']):
                if task not in outputs['col_idxs']:
                    continue  # when predicting only one set with --dataset

                task_idx = outputs['col_idxs'][task]
                if task_idx == -1:  # sent-level task, already covered
                    continue

                if task_type == 'string2string':
                    tok[task_idx] = apply_lemma_rule(outputs['tokens'][word_idx], outputs[task][word_idx])
                    if tok[task_idx] == '':
                        tok[task_idx] = outputs['tokens'][word_idx]
                elif task_type == 'dependency':
                    tok[task_idx] = str(outputs[task + '_head_indices'][word_idx])
                    if len(tok) - 1 == task_idx:
                        tok.append('')
                    tok[task_idx+1] = str(outputs[task + '_rels'][word_idx])
                elif task_type == 'multiseq':
                    tok[task_idx] = '|'.join(outputs[task][word_idx])
                elif task_type in ['seq', 'seq_bio']:
                    tok[task_idx] = outputs[task][word_idx]
                else:
                    logger.error("Error, task_type " + task_type + ' is not defined in predictor')
                    exit(1)
            
            output_lines.append('\t'.join(tok))            
        return '\n'.join(output_lines) + '\n\n'

    @overrides
    def dump_line(self, outputs: JsonDict) -> str:
        remove_idxs = []
        for task_idx, task in enumerate(outputs['tasks']):
            if task not in outputs and task + '_rels' not in outputs:
                remove_idxs.append(task_idx)
        for task_idx in reversed(remove_idxs):
            del outputs['tasks'][task_idx]
            del outputs['task_types'][task_idx]

        task_types = []
        for task in outputs['col_idxs']:
            if task == 'word_idx':
                continue
            task_idx = outputs['tasks'].index(task)
            task_types.append(outputs['task_types'][task_idx])

        # classification
        if sum([task_types.count(task) for task in ['classification', 'probdistr', 'regression']]) == len(task_types):
            return self.to_str_sentlevel(outputs)
        # generation
        elif 'seq2seq' in task_types:
            return self.to_str_seq2seq(outputs)
        # mlm
        elif 'mlm' in task_types:
            logger.error('Sorry, predicting with MLM is currently not supported, as it is unclear what it should do')
            return "" 
            #exit(1)
        # sequence labeling or both
        else:
            return self.to_str_wordlevel(outputs)
