import logging
import os

import torch

logger = logging.getLogger(__name__)

from machamp.model.machamp import MachampModel


class Callback():
    def __init__(self, keep_best_n: int = 1):
        """
        Class that keeps track of performance of models over epochs
        and handles model saving where necessary.

        Parameters
        ----------
        keep_best_n: int
            the amount of models to keep
        """
        self.keep_best_n = keep_best_n
        self.sum_scores = {}
        self.full_scores = {}

    def save_model(self,
                   epoch: int,
                   full_scores: dict,
                   model: MachampModel,
                   serialization_dir: str):
        """
        This function is registering a new model with its score. Despite its 
        name, it only saves the model if it belongs to the best_n models.

        Parameters
        ----------
        epoch: int
            The number of the epoch.
        full_scores: dict
            Dictionary of all score information of this epoch. We assume
            that there is a key 'sum' which we use for ranking.
        model: MachampModel
            The model to save if it belongs to the best_n.
        serialization_dir: str
            The folder where the models should be saved.
        """
        score = full_scores['sum']
        if self.keep_best_n == 0:
            return

        # assuming higher is better
        if len(self.sum_scores) < self.keep_best_n or self.sum_scores[
            sorted(self.sum_scores, key=self.sum_scores.get, reverse=True)[self.keep_best_n - 1]] < score:
            # save new model
            tgt_path = os.path.join(serialization_dir, 'model_' + str(epoch) + '.pt')
            logger.info("Performance of " + str(score) + ' within top ' + str(
                self.keep_best_n) + ' models, saving to ' + tgt_path)
            torch.save(model, tgt_path)

            # remove an old model if necessary
            if len(self.sum_scores) >= self.keep_best_n:
                epoch_to_remove = sorted(self.sum_scores, key=self.sum_scores.get, reverse=True)[self.keep_best_n - 1]
                path_to_remove = os.path.join(serialization_dir, 'model_' + str(epoch_to_remove) + '.pt')
                os.remove(path_to_remove)

        self.sum_scores[epoch] = score
        self.full_scores[epoch] = full_scores

    def get_best_epoch(self):
        """
        Finds the best performing epoch (based on sum).

        Returns
        -------
        epoch: int
            The epoch in which the best "sum" score is obtained
        """
        return sorted(self.sum_scores, key=self.sum_scores.get, reverse=True)[0]
        

    def link_model(self, serialization_dir: str, epoch: int):
        """
        Create a symbolic link of the model of the ebst epoch in 
        model.pt.

        Parameters
        ----------
        serialization_dir: str
            The folder where the models should be saved.
        """ 
        src = 'model_' + str(epoch) + '.pt'
        tgt = os.path.join(serialization_dir, 'model.pt')
        logger.info(
            "Best performance obtained in epoch " + str(epoch) + ' linking model ' + src + ' as ' + tgt + '.')
        os.symlink(src, tgt)

    def get_full_scores(self, epoch: int):
        """
        Returns all scores for a certain epoch.

        Parameters
        ----------
        epoch: int
            which epoch to find the scores for

        Returns
        -------
        scores: dict
            a dictionary containing score string and their values
        """
        return self.full_scores[epoch]
