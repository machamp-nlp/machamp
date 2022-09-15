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
        self.scores = {}

    def save_model(self,
                   epoch: int,
                   score: float,
                   model: MachampModel,
                   serialization_dir: str):
        """
        This function is registering a new model with its score. Despite its 
        name, it only saves the model if it belongs to the best_n models.

        Parameters
        ----------
        epoch: int
            The number of the epoch.
        score: float
            The score of the model at this epoch that we should 
            take into account (usually sum over tasks).
        model: MachampModel
            The model to save if it belongs to the best_n.
        serialization_dir: str
            The folder where the models should be saved.
        """
        if self.keep_best_n == 0:
            return

        # assuming higher is better
        if len(self.scores) < self.keep_best_n or self.scores[
            sorted(self.scores, key=self.scores.get, reverse=True)[self.keep_best_n - 1]] < score:
            # save new model
            tgt_path = os.path.join(serialization_dir, 'model_' + str(epoch) + '.pt')
            logger.info("Performance of " + str(score) + ' within top ' + str(
                self.keep_best_n) + ' models, saving to ' + tgt_path)
            torch.save(model, tgt_path)

            # remove an old model if necessary
            if len(self.scores) >= self.keep_best_n:
                epoch_to_remove = sorted(self.scores, key=self.scores.get, reverse=True)[self.keep_best_n - 1]
                path_to_remove = os.path.join(serialization_dir, 'model_' + str(epoch_to_remove) + '.pt')
                os.remove(path_to_remove)

        self.scores[epoch] = score

    def copy_best(self, serialization_dir: str):
        """
        Create a symbolic link of the model of the ebst epoch in 
        model.pt.

        Parameters
        ----------
        serialization_dir: str
            The folder where the models should be saved.
        """
        best_epoch = str(sorted(self.scores, key=self.scores.get, reverse=True)[0])
        src = 'model_' + str(best_epoch) + '.pt'
        tgt = os.path.join(serialization_dir, 'model.pt')
        logger.info(
            "Best performance obtained in epoch " + str(best_epoch) + ' linking model ' + src + ' as ' + tgt + '.')
        os.symlink(src, tgt)
        return best_epoch
