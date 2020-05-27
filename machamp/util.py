"""
A collection of handy utilities
"""

from typing import List, Tuple, Dict, Any

import os
import glob
import json
import logging
import tarfile
import traceback
import torch
import pprint
import copy
import numpy

from allennlp.common.checks import ConfigurationError
from allennlp.common import Params
from allennlp.common.params import with_fallback
from allennlp.commands.predict import _PredictManager
from allennlp.common.checks import check_for_gpu
from allennlp.models.archival import load_archive
from allennlp.predictors.predictor import Predictor

logger = logging.getLogger(__name__)

# count number of sentences in file, if it is a connlu-like
# file it counts the empty lines, otherwise it counts all 
# lines
def countLines(path):
    total = 0
    empty = 0
    for line in open(path):
        total += 1
        if line.strip() == '':
            empty += 1
    if empty < 10:
        return total
    else:
        return empty

def merge_configs(parameters_config: str, dataset_config: str, overrides: Dict) -> Params:
    """
    Merges a dataset config file with a parameters config file
    """
    mergedSettings = Params.from_file(parameters_config).as_dict()
    mergedSettings = with_fallback(overrides, mergedSettings)#.update(overrides)
    #mergedSettings =  Params(mergedSettings)
    dataset_config = Params.from_file(dataset_config)
    defaultDecoder = mergedSettings['model'].pop('default_decoder')
    orderedStuff = {}
    mergedSettings['dataset_reader']['datasets'] = {}
    mergedSettings['model']['decoders'] = {}

    for dataset in dataset_config:
        dataReader = {} 
        dataReader['train'] = dataset_config[dataset]['train_data_path']
        dataReader['dev'] = dataset_config[dataset]['validation_data_path']
        if 'test_data_path' in dataset_config[dataset]:
            dataReader['test'] = dataset_config[dataset]['test_data_path']

        if 'word_idx' in dataset_config[dataset]:
            dataReader['word_idx'] = dataset_config[dataset]['word_idx']
        else:
            dataReader['sent_idxs'] = dataset_config[dataset]['sent_idxs']
        
        dataReader['tasks'] = {}
        if 'copy_other_columns' in dataset_config[dataset]:
            dataReader['copy_other_columns'] = dataset_config[dataset]['copy_other_columns']
        else:
            dataReader['copy_other_columns'] = mergedSettings['model']['default_dataset']['copy_other_columns']

        for task in dataset_config[dataset]['tasks']:
            taskOverride = dataset_config[dataset]['tasks'][task]
            decoder = copy.deepcopy(defaultDecoder)
            decoder.update(taskOverride)

            decoder['dataset'] = dataset
            decoder['task'] = task

            dataReader['tasks'][task] = copy.deepcopy(decoder)
            orderIdx = decoder['order']
            if 'task_type' not in decoder:
                logger.warning('Error, task ' + task + ' has no defined task_type')
                exit(1)
            curTrans = decoder['task_type']
            curLayer = decoder['layer']
            

            if decoder['task_type'] == 'dependency':
                decoder['type'] = 'machamp_dependency_decoder'
                if 'metric' not in dataReader['tasks'][task]:
                    decoder['metric'] = 'LAS'
                if 'tag_representation_dim' not in dataReader['tasks'][task]:
                    decoder['tag_representation_dim'] = 256
                if 'arc_representation_dim' not in dataReader['tasks'][task]:
                    decoder['arc_representation_dim'] = 768

            elif decoder['task_type'] == 'classification':
                decoder['type'] = 'machamp_sentence_classifier'
                #ROB TODO  why do we need empty kwargs?
                decoder['kwargs'] = {}

            elif decoder['task_type'] == 'multiseq':
                decoder['type'] = 'multiseq_decoder'

            elif decoder['task_type'] in ['seq', 'string2string']:
                if 'decoder_type' in decoder and decoder['decoder_type'] == 'crf':
                    decoder['type'] = 'masked_crf_decoder'
                    del decoder['decoder_type']
                    del decoder['decoder_type']
                else:
                    decoder['type'] = 'machamp_tag_decoder'
            
            else: 
                logger.warning('task_type ' + str(dataReader['tasks'][task]['task_type']) + " not known")
                exit(1)

            if 'metric' not in decoder:
                decoder['metric'] = 'acc'
            if decoder['metric'] == 'span_f1':
                decoder['metric'] = 'machamp_span_f1'
            orderedStuff[task] = [orderIdx, curTrans, curLayer]

            # save stuff in mergedSettings
            mergedSettings['model']['decoders'][task] = decoder
            dataReader['tasks'][task] = copy.deepcopy(decoder)
        mergedSettings['dataset_reader']['datasets'][dataset] = dataReader
        # Rob: we definitely do not want to cheat and add dev and test labels here
        mergedSettings["datasets_for_vocab_creation"] = ["train"]
    
    del mergedSettings['model']['default_dataset']

    # to support reading from multiple files we add them to the datasetreader constructor instead
    # the following ones are there just here to make allennlp happy
    mergedSettings['train_data_path'] = 'train'
    mergedSettings['validation_data_path'] = 'dev'
    if 'test_data_path' in dataset_config[dataset]:
        mergedSettings['test_data_path'] = 'test'
    
    # generate ordered lists, which make it easier to use in the machamp model
    orderedTasks = []
    orderedTaskTypes = []
    orderedLayers = []
    for label, idx in sorted(orderedStuff.items(), key=lambda item: item[1]):
        orderedTasks.append(label)
        orderedTaskTypes.append(orderedStuff[label][1])
        orderedLayers.append(orderedStuff[label][2])
    mergedSettings['model']['tasks'] = orderedTasks
    mergedSettings['model']['task_types'] = orderedTaskTypes
    mergedSettings['model']['layers_for_tasks'] = orderedLayers
    
    mergedSettings['model']['decoders'][orderedTasks[0]]['prev_task'] = None
    for taskIdx, task in enumerate(orderedTasks[1:]):
        mergedSettings['model']['decoders'][task]['prev_task'] = orderedTasks[taskIdx] 
        #TODO shouldnt this be -1?
    for task in orderedTasks:
        mergedSettings['model']['decoders'][task]['task_types'] = orderedTaskTypes 
        mergedSettings['model']['decoders'][task]['tasks'] = orderedTasks 
        #taskIdx is not +1, because first item is skipped

    # remove items from tagdecoder, as they are not neccesary there
    for item in ['task_type', 'dataset', 'column_idx', 'layer', 'order']:
        for task in mergedSettings['model']['decoders']:
            if item in mergedSettings['model']['decoders'][task]:
                del mergedSettings['model']['decoders'][task][item]

    
    if 'trainer' in overrides and 'cuda_device' in overrides['trainer']:
        mergedSettings['trainer']['cuda_device'] = overrides['trainer']['cuda_device']
    #import pprint
    #pprint.pprint(mergedSettings.as_dict())
    #exit(1)
    numSents = 0
    for dataset in mergedSettings['dataset_reader']['datasets']:
        trainPath = mergedSettings['dataset_reader']['datasets'][dataset]['train']
        numSents += countLines(trainPath)
    warmup = int(numSents/mergedSettings['iterator']['batch_size'])
    mergedSettings['trainer']['learning_rate_scheduler']['warmup_steps'] = warmup
    mergedSettings['trainer']['learning_rate_scheduler']['start_step'] = warmup
    mergedSettings['model']['bert_path'] = mergedSettings['dataset_reader']['token_indexers']['bert']['pretrained_model']

    #TODO, this will result in the same as appending _tags , however, the 
    # warning will still be there... this can be circumvented by copying 
    # allennlp.data.fields.sequence_label_field and add a smarter check...
    #mergedSettings['vocabulary'] = {'non_padded_namespaces': ['ne1']}
    return Params(mergedSettings)


def predict_model_with_archive(predictor: str, params: Params, archive: str,
                               input_file: str, output_file: str, batch_size: int = 1):
    cuda_device = params["trainer"]["cuda_device"]

    check_for_gpu(cuda_device)
    archive = load_archive(archive,
                           cuda_device=cuda_device)
    for item in archive.config.duplicate():
        archive.config.__delitem__(item)
    for item in params:
        archive.config[item] = params.as_dict()[item]

    predictor = Predictor.from_archive(archive, predictor)

    manager = _PredictManager(predictor,
                              input_file,
                              output_file,
                              batch_size,
                              print_to_console=False,
                              has_dataset_reader=True)
    manager.run()


def predict_model(predictor: str, params: Params, archive_dir: str,
                  input_file: str, output_file: str, batch_size: int = 1):
    """
    Predict output annotations from the given model and input file and produce an output file.
    :param predictor: the type of predictor to use, e.g., "machamp_predictor"
    :param params: the Params of the model
    :param archive_dir: the saved model archive
    :param input_file: the input file to predict
    :param output_file: the output file to save
    :param batch_size: the batch size, set this higher to speed up GPU inference
    """
    archive = os.path.join(archive_dir, "model.tar.gz")
    predict_model_with_archive(predictor, params, archive, input_file, output_file, batch_size)


def cleanup_training(serialization_dir: str, keep_archive: bool = False, keep_weights: bool = False):
    """
    Removes files generated from training.
    :param serialization_dir: the directory to clean
    :param keep_archive: whether to keep a copy of the model archive
    :param keep_weights: whether to keep copies of the intermediate model checkpoints
    """
    if not keep_weights:
        for file in glob.glob(os.path.join(serialization_dir, "*.th")):
            os.remove(file)
    if not keep_archive:
        os.remove(os.path.join(serialization_dir, "model.tar.gz"))


def archive_bert_model(serialization_dir: str, config_file: str, output_file: str = None):
    """
    Extracts BERT parameters from the given model and saves them to an archive.
    :param serialization_dir: the directory containing the saved model archive
    :param config_file: the configuration file of the model archive
    :param output_file: the output BERT archive name to save
    """
    archive = load_archive(os.path.join(serialization_dir, "model.tar.gz"))


    model = archive.model
    model.eval()

    try:
        bert_model = model.text_field_embedder.token_embedder_bert.model
    except AttributeError:
        logger.warning(f"Could not find the BERT model inside the archive {serialization_dir}")
        traceback.print_exc()
        return

    weights_file = os.path.join(serialization_dir, "pytorch_model.bin")
    torch.save(bert_model.state_dict(), weights_file)

    if not output_file:
        output_file = os.path.join(serialization_dir, "bert-finetune.tar.gz")

    with tarfile.open(output_file, 'w:gz') as archive:
        archive.add(config_file, arcname="bert_config.json")
        archive.add(weights_file, arcname="pytorch_model.bin")

    os.remove(weights_file)


def to_multilabel_sequence(predictions, vocab, task):
    #TODO @AR: Hard-coded parameters for now
    THRESH = 0.5
    k = 2
    outside_index = vocab.get_token_index("O", namespace=task)

    # @AR: Get the thresholded matrix and prepare the prediction sequence
    pred_over_thresh = (predictions >= THRESH) * predictions
    sequence_token_labels = []

    # @AR: For each label set, check if to apply argmax or sigmoid thresh
    for pred in pred_over_thresh:
        num_pred_over_thresh = numpy.count_nonzero(pred)

        if num_pred_over_thresh < k:
            pred_idx_list = [numpy.argmax(predictions, axis=-1)]
            # print("argmax  ->", pred_idx_list)
        else:
            pred_idx_list = [numpy.argmax(predictions, axis=-1)]
            # pred_idx_list = list(numpy.argpartition(pred, -k)[-k:])
            # # print("sigmoid ->", pred_idx_list)

            # # If the first (i.e., second best) is "O", ignore/remove it
            # if pred_idx_list[0] == outside_index:
            #     pred_idx_list = pred_idx_list[1:]
            # # If the second (i.e., the best) is "O", ignore/remove the first
            # elif pred_idx_list[1] == outside_index:
            #     pred_idx_list = pred_idx_list[1:]
            # else:
            #     pass

        sequence_token_labels.append(pred_idx_list)

    return sequence_token_labels
