import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from builders.vocab_builder import build_vocab
from shutil import copyfile

from utils.logging_utils import setup_logger
from builders.model_builder import build_model

import os
import numpy as np
import pickle
import random

class BaseTask:
    def __init__(self, config):

        self.logger = setup_logger()

        self.checkpoint_path = os.path.join(config.training.checkpoint_path, config.model.name)
        if not os.path.isdir(self.checkpoint_path):
            self.logger.info("Creating checkpoint path")
            os.makedirs(self.checkpoint_path)

        if not os.path.isfile(os.path.join(self.checkpoint_path, "vocab.bin")):
            self.logger.info("Creating vocab")
            self.vocab = self.load_vocab(config.vocab)
            self.logger.info("Saving vocab to %s" % os.path.join(self.checkpoint_path, "vocab.bin"))
            pickle.dump(self.vocab, open(os.path.join(self.checkpoint_path, "vocab.bin"), "wb"))
        else:
            self.logger.info("Loading vocab from %s" % os.path.join(self.checkpoint_path, "vocab.bin"))
            self.vocab = pickle.load(open(os.path.join(self.checkpoint_path, "vocab.bin"), "rb"))

        self.logger.info("Loading data")
        self.load_datasets(config.dataset)
        self.create_dataloaders(config)

        self.logger.info("Building model")
        self.model = build_model(config.model, self.vocab)
        self.config = config
        self.device = torch.device(config.model.device)

        self.logger.info("Defining optimizer and objective function")
        self.configuring_hyperparameters(config)
        self.optim = Adam(self.model.parameters(), lr=config.training.learning_rate, betas=(0.9, 0.98))
        self.scheduler = LambdaLR(self.optim, self.lambda_lr)

    def configuring_hyperparameters(self, config):
        raise NotImplementedError

    def load_vocab(self, config):
        vocab = build_vocab(config)

        return vocab
    
    def load_datasets(self, config):
        raise NotImplementedError

    def create_dataloaders(self, config):
        raise NotImplementedError
    
    def compute_scores(self, inputs: torch.Tensor, labels: torch.Tensor) -> dict:
        raise NotImplementedError

    def evaluate_metrics(self, dataloader: DataLoader):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError

    def lambda_lr(self, step):
        warm_up = self.warmup
        step += 1
        return (self.model.d_model ** -.5) * min(step ** -.5, step * warm_up ** -1.5)

    def load_checkpoint(self, fname) -> dict:
        if not os.path.exists(fname):
            return None

        self.logger.info("Loading checkpoint from %s", fname)

        checkpoint = torch.load(fname, weights_only=False)

        torch.set_rng_state(checkpoint['torch_rng_state'])
        torch.cuda.set_rng_state(checkpoint['cuda_rng_state'])
        np.random.set_state(checkpoint['numpy_rng_state'])
        random.setstate(checkpoint['random_rng_state'])

        self.model.load_state_dict(checkpoint['state_dict'], strict=False)

        self.logger.info("Resuming from epoch %s", checkpoint['epoch'])

        return checkpoint

    def save_checkpoint(self, dict_for_updating: dict) -> None:
        dict_for_saving = {
            **dict_for_updating,
            'torch_rng_state': torch.get_rng_state(),
            'cuda_rng_state': torch.cuda.get_rng_state(),
            'numpy_rng_state': np.random.get_state(),
            'random_rng_state': random.getstate()
        }

        torch.save(dict_for_saving, os.path.join(self.checkpoint_path, "last_model.pth"))

    def start(self):
        if os.path.isfile(os.path.join(self.checkpoint_path, "last_model.pth")):
            checkpoint = self.load_checkpoint(os.path.join(self.checkpoint_path, "last_model.pth"))
            best_score = checkpoint["best_score"]
            patience = checkpoint["patience"]
            self.epoch = checkpoint["epoch"] + 1
            self.optim.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
        else:
            best_score = .0
            patience = 0

        while True:
            self.train()
            # val scores
            scores, _ = self.evaluate_metrics(self.dev_dataloader)
            self.logger.info("Validation scores %s", scores)
            score = scores[self.score]

            # Prepare for next epoch
            is_the_best_model = False
            if score > best_score:
                best_score = score
                patience = 0
                is_the_best_model = True
            else:
                patience += 1

            exit_train = False

            if patience == self.patience:
                self.logger.info('patience reached.')
                exit_train = True

            self.save_checkpoint({
                "epoch": self.epoch,
                "best_score": best_score,
                "patience": patience,
                "state_dict": self.model.state_dict(),
                "optimizer": self.optim.state_dict(),
                "scheduler": self.scheduler.state_dict()
            })

            if is_the_best_model:
                copyfile(
                    os.path.join(self.checkpoint_path, "last_model.pth"), 
                    os.path.join(self.checkpoint_path, "best_model.pth")
                )

            if exit_train:
                break

            self.epoch += 1

    def get_predictions(self, dataset, get_scores=True):
        raise NotImplementedError
