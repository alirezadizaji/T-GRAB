import torch
import os
from typing import Any, Dict, List

from ...DTDG.link_pred.trainer import LinkPredTrainer
from ...DTDG.trainer import DTDGTrainer

class IdentityModel(torch.nn.Module):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def forward(self, x):
        return x

class LinkPredEmptyTrainer(LinkPredTrainer):
    def __init__(self):
        DTDGTrainer.__init__(self)

    def get_model(self):
        return {'node_emb': IdentityModel()}

    def get_optimizer(self) -> torch.optim.Optimizer:
        return NotImplementedError("Empty model does not support getting optimizer.")
    
    def get_criterion(self) -> torch.nn.Module:
        return NotImplementedError("Empty model does not support getting criterion.")
    
    def before_training(self):
        pass

    def before_starting_window_training(self):
        pass
    
    def train_for_one_epoch(self, split_mode: str):
        raise NotImplementedError("Empty model does not support training.")

    def early_stopping_checker(self, early_stopper) -> bool:
        raise NotImplementedError("Empty model does not support early stopping.")
    
    def forward_backbone(self, snapshot: torch.Tensor, snapshot_feat: torch.Tensor, node_feat: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("Empty model does not support forward backbone.")
    
    def update_metrics(self, 
                        curr_snapshot: torch.Tensor, 
                        snapshot_t: int, 
                        snapshot_idx: int,
                        metrics_list: Dict[str, List[Any]], 
                        split_mode: str):
        raise NotImplementedError("Empty model does not support updating metrics.")

    
    def get_dataset_regex_pattern(self) -> str:
        """ Each task in link prediction supports a group of datasets that follow a regex pattern. To understand current supported tasks,
        please checkout subfolders of `link_pred`"""
        raise NotImplementedError("Empty model does not support getting dataset regex pattern.")

    def _get_run_save_dir(self) -> str:
        linkpredtrainer_run_save_dir = super(LinkPredEmptyTrainer, self)._get_run_save_dir()
        return ("/".join(linkpredtrainer_run_save_dir.split("/")[:-3]) + \
                        "/empty/" + 
                        "/".join(linkpredtrainer_run_save_dir.split("/")[-2:]))

    def list_of_metrics_names(self) -> List[str]:
        raise NotImplementedError("Empty model does not support getting list of metrics names.")