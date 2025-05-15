from abc import abstractmethod
from argparse import ArgumentParser
from typing import Dict
import os

import numpy as np
import torch

from TSA.train.edgebank import Trainer
from ...model.edgebank_predictor import EdgeBankPredictor

class EdgeBankTrainer(Trainer):
    def __init__(self):
        super(EdgeBankTrainer, self).__init__()
        # Set model
        self.model = self.get_model()
        
    @staticmethod
    def _set_running_args(parser: ArgumentParser) -> ArgumentParser:
        """ Appending new arguments that are being used by Discrete-Time Dynamic graph tasks. """
        parser = Trainer._set_running_args(parser)
        parser.add_argument('-d', '--data', type=str, help='Dataset name')
        parser.add_argument('-l', '--data-loc', type=str, help='The location where data is stored.')

        return parser

    def set_model_args(self, parser: ArgumentParser) -> ArgumentParser:
        parser.add_argument('--mem_mode', type=str, help='Memory mode', default='unlimited', choices=['unlimited', 'fixed_time_window'])
        parser.add_argument('--time_window_ratio', type=float, help='Test window ratio', default=0.15)

        return parser

    def get_model(self) -> Dict[str, torch.nn.Module]:
        """ Fill your model by calling this function """
        edgebank = EdgeBankPredictor(np.array(self.train_loader.dataset.src),
                                     np.array(self.train_loader.dataset.dst),
                                     np.array(self.train_loader.dataset.t),
                                     memory_mode=self.args.mem_mode,
                                     time_window_ratio=self.args.time_window_ratio)

        
        return {'edgebank': edgebank}

    def _get_run_save_dir(self) -> str:
        trainer_run_save_dir = super(EdgeBankTrainer, self)._get_run_save_dir()
        
        return os.path.join(
            trainer_run_save_dir, 
            "CTDG",
            "EdgeBank")
