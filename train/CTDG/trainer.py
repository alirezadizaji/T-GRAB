from abc import abstractmethod
from argparse import ArgumentParser
from typing import Dict
import os

import torch

from TSA.train.trainer import Trainer
from ...utils import  NodeFeatType, verify_kwd_existence

NODE_EMB_MODEL_NAME = "node_emb"
class CTDGTrainer(Trainer):
    def __init__(self):
        super(CTDGTrainer, self).__init__()
        # Set model
        
        self.model = self._wrapper_to_verify_and_return_model()
        
        print("\n ============ Number of Parameters =============")
        for name_, model_ in self.model.items():
            num_params = sum(p.numel() for p in model_.parameters() if p.requires_grad)
            print(f"$$ Model: {name_}, number of params: {num_params}", flush=True)

        # Set training optimization
        self.optim = self.get_optimizer()

        # Set training objective
        self.criterion = self.get_criterion()

        
    @staticmethod
    def _set_running_args(parser: ArgumentParser) -> ArgumentParser:
        """ Appending new arguments that are being used by Discrete-Time Dynamic graph tasks. """
        parser = Trainer._set_running_args(parser)
        parser.add_argument('-d', '--data', type=str, help='Dataset name')
        parser.add_argument('--node-feat', choices=NodeFeatType.list(), help='Type of node feature generation', default=NodeFeatType.CONSTANT)
        parser.add_argument('--node-feat-dim', type=int, default=1, help='Number of dimension of node features. Being used by `CONSTANT`, `RAND`, and `RANDN` node feature types.')
        parser.add_argument('-l', '--data-loc', type=str, help='The location where data is stored.')

        return parser

    @verify_kwd_existence(NODE_EMB_MODEL_NAME)
    def _wrapper_to_verify_and_return_model(self) -> Dict[str, torch.nn.Module]:
        return self.get_model()

    def get_model(self) -> Dict[str, torch.nn.Module]:
        """ Fill your model by calling this function. The function should return a dictionary that necessarily contains a model with keyword `node_emb`."""
        return {}

    @abstractmethod
    def get_optimizer(self) -> torch.optim.Optimizer:
        pass
    
    @abstractmethod
    def get_criterion(self) -> torch.nn.Module:
        pass

    def _get_model_card(self) -> str:
        """ In Discrete-time dynamic graph training, type of node feature and its dimension is considered within the model card."""
        trainer_get_model_id = super(CTDGTrainer, self)._get_model_card()
        return f'nodefeat={self.args.node_feat}_nodedim={self.args.node_feat_dim}_' + trainer_get_model_id
    
    def _get_run_save_dir(self) -> str:
        trainer_run_save_dir = super(CTDGTrainer, self)._get_run_save_dir()
        
        return os.path.join(
            trainer_run_save_dir, 
            "CTDG")
