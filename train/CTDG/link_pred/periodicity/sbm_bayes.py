import argparse
import sys
import re

import torch
import os
from typing import Any, Dict, List

from ....CTDG.link_pred.trainer import LinkPredTrainer
from .....model.node_emb import NodeEmbeddingModel
from .....dataset.DTDG.graph_generation.periodicity import PeriodicGenerator

class IdentityModel(NodeEmbeddingModel):
    def __init__(self):
        super(IdentityModel, self).__init__()

    def forward(self, x):
        return x

class PeriodicSBMBayesTrainer(LinkPredTrainer):
    def __init__(self):
        super(PeriodicSBMBayesTrainer, self).__init__()

        icp_match = re.search(r"-icp([\d.]+)", self.args.data)
        incp_match = re.search(r"-incp([\d.]+)", self.args.data)

        if icp_match and incp_match:
            self.intra_cluster_prob = float(icp_match.group(1))
            self.inter_cluster_prob = float(incp_match.group(1))
            print(f"Intra-cluster probability: {self.intra_cluster_prob}")
            print(f"Inter-cluster probability: {self.inter_cluster_prob}")
        else:
            raise Exception("Could not extract probabilities.")

    def get_model(self):
        return {'node_emb': IdentityModel(), 'link_pred': IdentityModel()}

    def get_optimizer(self) -> torch.optim.Optimizer:
        return NotImplementedError("SBM Bayes model does not support getting optimizer.")
    
    def get_criterion(self) -> torch.nn.Module:
        return NotImplementedError("SBM Bayes model does not support getting criterion.")
    
    def before_training(self):
        pass

    def before_starting_window_training(self):
        pass
    
    def train_for_one_epoch(self, split_mode: str):
        raise NotImplementedError("SBM Bayes model does not support training.")

    def early_stopping_checker(self, early_stopper) -> bool:
        raise NotImplementedError("SBM Bayes model does not support early stopping.")
    
    def forward_backbone(self, snapshot: torch.Tensor, snapshot_feat: torch.Tensor, node_feat: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError("SBM Bayes model does not support forward backbone.")
    
    def after_epoch_evaluation(self, split_mode: str):
        raise NotImplementedError("SBM Bayes model does not support after epoch evaluation.")
    
    def after_epoch_training(self):
        raise NotImplementedError("SBM Bayes model does not support after epoch training.")
    
    def after_iteration_evaluation(self, split_mode):
        raise NotImplementedError("SBM Bayes model does not support after iteration evaluation.")
    
    def before_epoch_evaluation(self, split_mode: str):
        raise NotImplementedError("SBM Bayes model does not support before epoch evaluation.")
    
    def before_epoch_training(self):
        raise NotImplementedError("SBM Bayes model does not support before epoch training.")

    def after_iteration_training(self):
        raise NotImplementedError("SBM Bayes model does not support after iteration training.")
    
    def set_model_args(self, parser):
        pass

    @property
    def model_params(self) -> List[str]:
        return []

    def get_dataset_regex_pattern(self):
        dataset_regex_pattern = PeriodicGenerator.PERIODIC_REGEX
        dataset_name_part_to_check = self.args.data.split("/")[0]
        return dataset_regex_pattern, dataset_name_part_to_check
   
    def _get_args(self) -> argparse.Namespace:
        parser = argparse.ArgumentParser('*** TSA ***', add_help=False)
        parser = self._set_running_args(parser)

        try:
            args = parser.parse_args()
        except:
            parser.print_help()
            sys.exit(0)
            
        return args

    def update_metrics(self, 
                        curr_snapshot: torch.Tensor, 
                        snapshot_t: int, 
                        snapshot_idx: int,
                        metrics_list: Dict[str, List[Any]], 
                        split_mode: str):
        raise NotImplementedError("SBM Bayes model does not support updating metrics.")

    
    def _get_run_save_dir(self) -> str:
        linkpredtrainer_run_save_dir = super(PeriodicSBMBayesTrainer, self)._get_run_save_dir()
        return ("/".join(linkpredtrainer_run_save_dir.split("/")[:-4]) + \
                        "/sbm_bayes/" + linkpredtrainer_run_save_dir.split("/")[-4] + "/" + \
                        "/".join(linkpredtrainer_run_save_dir.split("/")[-3:]))

    def list_of_metrics_names(self) -> List[str]:
        raise NotImplementedError("SBM Bayes model does not support getting list of metrics names.")

    def eval_for_one_epoch(self, split_mode: str):
        assert not torch.is_grad_enabled(), "During evaluation, torch grad should be disabled."

        community_2 = (2 * self.intra_cluster_prob) / (1 + self.intra_cluster_prob + 50 * self.inter_cluster_prob / 49)
        community_10 = (2 * self.intra_cluster_prob) / (1 + self.intra_cluster_prob + 10 * self.inter_cluster_prob)
        avg_f1 = (community_2 + community_10) / 2
        metrics_list = {
            "avg_f1": avg_f1,
        }
        
        return metrics_list 