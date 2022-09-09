import datetime
import json
import logging
import os
import random
import sys
from typing import List

import torch
import transformers
from torch.utils.data import DataLoader
from tqdm import tqdm
from uniplot import plot_to_string

from machamp.utils import myutils
from machamp.model.machamp import MachampModel
from machamp.model.callback import Callback
from machamp.data.machamp_dataset import MachampDataset
from machamp.utils import image
from machamp.data.machamp_sampler import MachampBatchSampler
from machamp.modules.allennlp.slanted_triangular import SlantedTriangular
from machamp.predictor.predict import predict


def train(
        name: str,
        parameters_config_path: str,
        dataset_config_paths: List[str],
        device: str,
        resume: str = None,
        retrain: str = None,
        seed: int = 8446,
        cmd: str = ''):
    """
    
    Parameters
    ----------
    name: str
        The name of the model.
    parameters_config_path: str
        Path to the hyperparameters configuration.
    dataset_config_paths: List[str]
        List of paths to dataset configurations.
    device: str
        Description of cuda device to use, i.e.: "cpu" or "gpu:0".
    resume: str = None
        Resume training of an incompleted training.
    retrain: str = None
        If retraining from a machamp model instead of 
        a transformers model, this holds the path to the
        previous model to use.
    seed: int = 8446
        Random seed, which is used for torch and the 
        random package. 
    cmd: str = ''
        The command invoked to start the training
    """
    start_time = datetime.datetime.now()
    if resume:  # TODO make this work
        parameters_config = myutils.load_json(resume + '/params-config.json')
        dataset_configs = myutils.load_json(resume + '/dataset-configs.json')
        serialization_dir = resume
    else:
        parameters_config = myutils.load_json(parameters_config_path)
        dataset_configs = myutils.merge_configs(dataset_config_paths, parameters_config)
        serialization_dir = 'logs/' + name + '/' + start_time.strftime("%Y.%m.%d_%H.%M.%S") + '/'
        counter = 2
        while os.path.isdir(serialization_dir):
            serialization_dir += '.' + str(counter)
            counter += 1
        os.makedirs(serialization_dir)

        if seed != None:
            parameters_config['random_seed'] = seed

        json.dump(dataset_configs, open(serialization_dir + '/dataset-config.json', 'w'), indent=4)
        json.dump(parameters_config, open(serialization_dir + '/params-configs.json', 'w'), indent=4)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        level=logging.INFO, handlers=[logging.FileHandler(os.path.join(serialization_dir, 'log.txt')),
                                                      logging.StreamHandler(sys.stdout)])
    logger = logging.getLogger(__name__)
    # sys.stdout = myutils.StreamToLogger(logger,logging.INFO)
    # sys.stderr = myutils.StreamToLogger(logger,logging.ERROR)

    if cmd != '':
        logger.info('cmd: ' + cmd)
    random.seed(parameters_config['random_seed'])
    torch.manual_seed(parameters_config['random_seed'])

    batch_size = parameters_config['batching']['batch_size']
    train_dataset = MachampDataset(parameters_config['transformer_model'], dataset_configs, is_train=True,
                                  max_input_length=parameters_config['encoder']['max_input_length'])
    train_sampler = MachampBatchSampler(train_dataset, batch_size, parameters_config['batching']['max_tokens'], True,
                                       parameters_config['batching']['sampling_smoothing'],
                                       parameters_config['batching']['sort_by_size'])
    train_dataloader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=lambda x: x)
    # Note that the vocabulary is only saved for debugging purposes, there is also a copy in the model.pt
    train_dataset.vocabulary.save_vocabs(os.path.join(serialization_dir, 'vocabularies'))

    dev_dataset = MachampDataset(parameters_config['transformer_model'], dataset_configs, is_train=False,
                                vocabulary=train_dataset.vocabulary,
                                max_input_length=parameters_config['encoder']['max_input_length'])
    dev_sampler = MachampBatchSampler(dev_dataset, batch_size, parameters_config['batching']['max_tokens'], True, 1.0,
                                     parameters_config['batching']['sort_by_size'])
    dev_dataloader = DataLoader(dev_dataset, batch_sampler=dev_sampler, collate_fn=lambda x: x)

    callback = Callback(parameters_config['training']['keep_top_n'])

    model = MachampModel(train_dataset.vocabulary, train_dataset.tasks, train_dataset.task_types,
                         parameters_config['transformer_model'], device, dataset_configs, train_dataset.tokenizer,
                         **parameters_config['encoder'])
    model.to(device)

    # This  makes the use of regexes not so usefull anymore, but we need to
    # extract the decoder attributes from the model (for MLM), and I wasnt sure how
    # to do it more elegantly
    first_group = []
    second_group = ['^decoders.*']
    pred_head_names = ["pred_layer", "cls", "lm_head", "generator_lm_head", "predictions", "mlm", "vocab_projector"]
    for attribute in model.named_parameters():
        if attribute[0].startswith('mlm'):
            if attribute[0].split('.')[1] in pred_head_names:
                second_group.append(attribute[0])
            elif 'decoder' in attribute[
                0]:  # for seq2seq models, this is a crude guess...., but if the second group is empty it crashes
                second_group.append(attribute[0])
            else:
                first_group.append(attribute[0])
    # first group contains MLM
    # second group contains all decoder heads
    parameter_groups = [[first_group, {}], [second_group, {}]]
    parameter_groups = myutils.make_parameter_groups(model.named_parameters(), parameter_groups)

    optimizer = transformers.AdamW(parameter_groups, **parameters_config['training']['optimizer'])
    scheduler = SlantedTriangular(optimizer, parameters_config['training']['num_epochs'], len(train_dataloader),
                                  **parameters_config['training']['learning_rate_scheduler'])

    start_training_time = datetime.datetime.now()
    logger.info("MaChAmp succesfully initialized in {:.1f}s".format((start_training_time - start_time).seconds))
    logger.info(image.machamp)
    logger.info('starting training...')
    all_dev_scores = {}

    for epoch in range(1, parameters_config['training']['num_epochs'] + 1):
        logger.info('Epoch ' + str(epoch) + ': training')
        epoch_start_time = datetime.datetime.now()
        model.train()
        model.reset_metrics()
        epoch_loss = 0.0

        for train_batch_idx, batch in enumerate(tqdm(train_dataloader, file=sys.stdout)):
            optimizer.zero_grad()
            # we create the batches again every epoch to save 
            # gpu ram, it is quite fast anyways
            batch = myutils.prep_batch(batch, device, train_dataset)

            loss, _, _, _ = model.forward(batch['token_ids'], batch['golds'], batch['seg_ids'], batch['eval_mask'],
                                          batch['offsets'], batch['subword_mask'])
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        scheduler.step()
        train_metrics = model.get_metrics()
        logger.info('Epoch ' + str(epoch) + ': evaluating on dev')
        model.eval()
        model.reset_metrics()
        dev_loss = 0.0
        dev_metrics = {}
        dev_batch_idx = 1
        if len(dev_dataset) > 0:

            for dev_batch_idx, batch in enumerate(tqdm(dev_dataloader, file=sys.stdout)):
                batch = myutils.prep_batch(batch, device, train_dataset)
                loss, _, _, _ = model.forward(batch['token_ids'], batch['golds'], batch['seg_ids'], batch['eval_mask'],
                                              batch['offsets'], batch['subword_mask'])
                dev_loss += loss.item()

            dev_metrics = model.get_metrics()
            callback.save_model(epoch, dev_metrics['sum'], model, serialization_dir)
        else:
            # use epoch number as metric, hack to always keep last model (as higher=better)
            callback.save_model(epoch, epoch, model, serialization_dir)

        if dev_batch_idx == 0:
            dev_batch_idx = 1
        if train_batch_idx == 0:
            train_batch_idx = 1
        info_dict = myutils.report_epoch(epoch_loss / train_batch_idx, dev_loss / dev_batch_idx, epoch, train_metrics,
                                         dev_metrics, epoch_start_time, start_training_time)
        json.dump(info_dict, open(os.path.join(serialization_dir, 'metrics_epoch_' + str(epoch) + '.json'), 'w'),
                  indent=4)

        # plot graph, should maybe be moved to callback?
        if 'sum' in dev_metrics:
            outlier = False
            if epoch > 4:
                mean = sum(all_dev_scores['sum']) / len(all_dev_scores['sum'])
                stdev = torch.std(torch.tensor(all_dev_scores['sum'])).item()
                dist = mean - all_dev_scores['sum'][0]
                outlier = dist > stdev

            for metric in dev_metrics:
                if metric not in all_dev_scores:
                    all_dev_scores[metric] = []
                all_dev_scores[metric].append(dev_metrics[metric])
            x = []
            mins = []
            for metric in sorted(all_dev_scores):
                if metric != 'sum':
                    x.append(all_dev_scores[metric])
                    if epoch > 4:
                        mins.append(min(all_dev_scores[metric][1:]))
            labels = [label for label in sorted(all_dev_scores) if label != 'sum']
            if outlier:
                plot = plot_to_string(x, title='Dev scores (x) over epochs (y)', legend_labels=labels, lines=True,
                                      y_min=min(mins))
            else:
                plot = plot_to_string(x, title='Dev scores (x) over epochs (y)', legend_labels=sorted(all_dev_scores),
                                      lines=True)
            logger.info('\n' + '\n'.join(plot))

    json.dump(info_dict, open(os.path.join(serialization_dir, 'metrics.json'), 'w'), indent=4)
    callback.copy_best(serialization_dir)

    if len(dev_dataloader.dataset.datasets) > 1:
        logger.info('Predicting on dev sets')
    else:
        logger.info('Predicting on dev set')

    if len(dev_dataset) > 0:
        # We have to re-read the dataset, because the old one might be shuffled (this happens in place in the sampler)
        dev_dataset = MachampDataset(parameters_config['transformer_model'], dataset_configs, is_train=False,
                                    vocabulary=train_dataset.vocabulary)
        dev_sampler = MachampBatchSampler(dev_dataset, batch_size, parameters_config['batching']['max_tokens'], False,
                                         1.0, False)
        dev_dataloader = DataLoader(dev_dataset, batch_sampler=dev_sampler, collate_fn=lambda x: x)
        # TODO use best model, not last!
        predict(model, dev_dataloader, serialization_dir, dataset_configs, train_dataset.tokenizer.sep_token_id,
                batch_size, device, train_dataset.vocabulary)
