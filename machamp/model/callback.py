import datetime
import json
import logging
import os

import torch

logger = logging.getLogger(__name__)

from machamp.model.machamp import MachampModel


class Callback:
    def __init__(self, serialization_dir, num_epochs, keep_best_n: int = 1):
        """
        Class that keeps track of performance of models over epochs
        and handles model saving where necessary.

        Parameters
        ----------
        keep_best_n: int
            the amount of models to keep
        """
        self.keep_best_n = keep_best_n
        self.serialization_dir = serialization_dir
        self.num_epochs = num_epochs
        self.start_time = datetime.datetime.now()
        self.epoch_start_time = datetime.datetime.now()
        # main metric to rank on:
        self.sums = {}
        # all metrics:
        self.dev_scores = []
        self.train_scores = []
        self.dev_losses = []
        self.train_losses = []

    def start_epoch_timer(self):
        self.epoch_start_time = datetime.datetime.now()

    def add_train_results(self, epoch, losses, metrics):
        self.train_scores.append(metrics)
        self.train_losses.append(losses)

    def add_dev_results(self, epoch, losses, metrics):
        self.dev_scores.append(metrics)
        self.dev_losses.append(losses)
        self.sums[epoch] = metrics['sum']

    def end_epoch(self, cur_epoch, model):
        if len(self.dev_scores) == 0:
            self.sums[cur_epoch] = cur_epoch
        # Find best epoch
        if cur_epoch != len(self.train_scores):
            logger.error('mismatch between epochs in callback and model')
            exit(1)

        if len(self.sums) == 0:
            best_epoch = cur_epoch
        else:
            best_epoch = sorted(self.sums, key=self.sums.get, reverse=True)[0]

        # Report scores (and write them)
        self.report_scores(cur_epoch, best_epoch)

        # Print graph
        if len(self.dev_scores) > 0:
            self.plot(cur_epoch)

        # Save model if necessary:
        self.save_model(cur_epoch, model)

        # Link to best model if this is the last epoch
        if cur_epoch == self.num_epochs:
            self.link_model(best_epoch)

    def report_scores(self, cur_epoch, best_epoch):
        # Start with meta-info:
        info = {'epoch': str(cur_epoch) + '/' + str(self.num_epochs), 'best_epoch': best_epoch}
        # Because the list with scores start at 0, and the epochs start
        # to count at 1
        best_epoch -= 1
        cur_epoch -= 1
        if torch.cuda.is_available():
            info['max_gpu_mem'] = torch.cuda.max_memory_allocated() * 1e-09

        _proc_status = '/proc/%d/status' % os.getpid()
        if os.path.isfile(_proc_status):
            data = open(_proc_status).read()
            i = data.index('VmRSS:')
            info['cur_ram'] = int(data[i:].split(None, 3)[1]) * 1e-06
        else:
            info['cur_ram'] = 0
        info['time_epoch'] = str(datetime.datetime.now() - self.epoch_start_time).split('.')[0]
        info['time_total'] = str(datetime.datetime.now() - self.start_time).split('.')[0]
        for key in info:
            if type(info[key]) == float:
                info[key] = '{:.4f}'.format(info[key])
        longest_key = max([len(key) for key in info]) + 1
        for key, value in info.items():
            logger.info(key + ' ' * (longest_key - len(key)) + ': ' + str(value))
        logger.info('')

        # extract all results
        table = [['', 'train_loss', 'dev_loss', 'train_scores', 'dev_scores']]
        for epoch_name, epoch in zip(['Best (' + str(best_epoch + 1) + ')', 'Epoch ' + str(cur_epoch + 1)],
                                     [best_epoch, cur_epoch]):
            table.append([epoch_name, '', '', '', ''])
            for item in sorted(self.train_scores[0]):
                prefix = '' if epoch != 0 else 'best_'
                if item == 'sum':
                    continue
                else:
                    task_name = item[:item.rfind('-')]
                    info[prefix + 'train_' + item] = self.train_scores[epoch][item]
                    info[prefix + 'train_' + task_name + '_loss'] = self.train_losses[epoch][task_name]
                    if len(self.dev_scores) > 0 and item in self.dev_scores[epoch]:
                        info[prefix + 'dev_' + item] = self.dev_scores[epoch][item]
                        info[prefix + 'dev_' + task_name + '_loss'] = self.dev_losses[epoch][task_name]
                        table.append([item, self.train_losses[epoch][task_name], self.dev_losses[epoch][task_name],
                                      self.train_scores[epoch][item], self.dev_scores[epoch][item]])
                    else:
                        table.append(
                            [item, self.train_losses[epoch][task_name], '-', self.train_scores[epoch][item], '-'])
            if len(self.dev_scores) > 0:
                table.append(['sum', self.train_losses[epoch]['sum'], self.dev_losses[epoch]['sum'],
                              self.train_scores[epoch]['sum'], self.dev_scores[epoch]['sum']])
            else:
                table.append(['sum', self.train_losses[epoch]['sum'], '-', self.train_scores[epoch]['sum'], '-'])

        # Print the table with all results
        for row_idx in range(len(table)):
            for cell_idx in range(len(table[row_idx])):
                if type(table[row_idx][cell_idx]) == float:
                    table[row_idx][cell_idx] = '{:.4f}'.format(table[row_idx][cell_idx])
        maxes = []
        for columnIdx in range(len(table[0])):
            maxes.append(max([len(row[columnIdx]) for row in table]))
        for row in table:
            row_str = ''
            for columnIdx, cell in enumerate(row):
                spacing = ' ' * (maxes[columnIdx] - len(cell))
                if columnIdx == 0:
                    row_str += cell + spacing + ' '
                else:
                    row_str += spacing + cell + ' '
            logger.info(row_str)

        info_ordered = {}
        for item in info:
            if 'dev' not in item and 'train' not in item:
                info_ordered[item] = info[item]
        for item in info:
            if 'loss' in item and 'train' in item:
                info_ordered[item] = info[item]
        for item in info:
            if 'loss' in item and 'dev' in item:
                info_ordered[item] = info[item]
        for item in info:
            if 'train' in item and 'loss' not in item:
                info_ordered[item] = info[item]
        for item in info:
            if 'dev' in item and 'loss' not in item:
                info_ordered[item] = info[item]

        # write all results to disk
        json.dump(info_ordered,
                  open(os.path.join(self.serialization_dir, 'metrics_epoch_' + str(cur_epoch) + '.json'), 'w'),
                  indent=4)
        if cur_epoch+1 == self.num_epochs:
            json.dump(info_ordered, open(os.path.join(self.serialization_dir, 'metrics.json'), 'w'), indent=4)

    def plot(self, epoch):
        try:
            from uniplot import plot_to_string
        except ImportError:
            logger.info("uniplot is not installed, so the results are not plotted.")
            return
        # We try to adjust the y-axis after epoch 5, because epoch 1 is often
        # extremely low.
        outlier = False
        if epoch > 4:
            mean = sum(self.sums.values()) / len(self.sums)
            stdev = torch.std(torch.tensor(list(self.sums.values())))
            dist = abs(mean - self.sums[1])
            outlier = dist > stdev

        x = []
        mins = []
        for metric in sorted(self.dev_scores[0]):
            if metric != 'sum':
                x.append([x[metric] for x in self.dev_scores])
                if epoch > 4:
                    mins.append(min(x[-1][1:]))

        labels = [label for label in sorted(self.dev_scores[0]) if label != 'sum']
        if outlier:
            plot = plot_to_string(x, title='Dev scores (y) over epochs (x)', legend_labels=labels, lines=True,
                                  y_min=min(mins))
        else:
            plot = plot_to_string(x, title='Dev scores (y) over epochs (x)', legend_labels=labels, lines=True)
        logger.info('\n' + '\n'.join(plot))

    def save_model(self,
                   epoch: int,
                   model: MachampModel):
        """
        This function is registering a new model with its score. Despite its 
        name, it only saves the model if it belongs to the best_n models.

        Parameters
        ----------
        epoch: int
            The number of the epoch.
        model: MachampModel
            The model to save if it belongs to the best_n.
        """
        best_n = sorted(self.sums, key=self.sums.get, reverse=True)[:self.keep_best_n]
        if epoch in best_n:
            tgt_path = os.path.join(self.serialization_dir, 'model_' + str(epoch) + '.pt')
            logger.info("Performance of {:.4f}".format(self.sums[epoch]) + ' within top ' + str(
                self.keep_best_n) + ' models, saving to ' + tgt_path)
            torch.save(model, tgt_path)
            if len(self.sums) > self.keep_best_n:
                epoch_to_remove = sorted(self.sums, key=self.sums.get, reverse=True)[self.keep_best_n]
                path_to_remove = os.path.join(self.serialization_dir, 'model_' + str(epoch_to_remove) + '.pt')
                if os.path.isfile(path_to_remove):
                    logger.info("Removing old model: " + path_to_remove)
                    os.remove(path_to_remove)

    def link_model(self, epoch: int):
        """
        Create a symbolic link of the model of the ebst epoch in 
        model.pt.

        Parameters
        ----------
        epoch: int
            epoch to link from.
        """
        src = 'model_' + str(epoch) + '.pt'
        tgt = os.path.join(self.serialization_dir, 'model.pt')
        logger.info(
            "Best performance obtained in epoch " + str(epoch) + ' linking model ' + src + ' as ' + tgt + '.')
        os.symlink(src, tgt)
