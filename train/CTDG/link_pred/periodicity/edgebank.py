import pickle
from typing import Any, Dict, List, Tuple

import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch

from ....CTDG.link_pred.edgebank import LinkPredEdgeBankTrainer
from .....dataset.DTDG.graph_generation.periodicity import PeriodicGenerator


class EdgeBankPeriodicTrainer(LinkPredEdgeBankTrainer):
    """ This class is only useful to train periodic-based datasets. 
    For more info about periodic datasets, please checkout the documentation of `PeriodicGenerator` class.
    """
        
    def get_dataset_regex_pattern(self):
        dataset_regex_pattern = PeriodicGenerator.PERIODIC_REGEX
        dataset_name_part_to_check = self.args.data.split("/")[0]
        return dataset_regex_pattern, dataset_name_part_to_check
   
    def list_of_metrics_names(self) -> List[str]:
        return ["avg_precision", "avg_recall", "avg_f1", "avg_acc"]

    def _get_graph_mask_ground_truth_and_pred(self, graph_mask, curr_snapshot):
        graph_src_indices, graph_dst_indices = torch.nonzero(graph_mask, as_tuple=True)
        graph_ground_truth = curr_snapshot[graph_src_indices, graph_dst_indices].float()
        graph_prediction = self.model['edgebank'].predict_link(graph_src_indices.cpu().numpy(), graph_dst_indices.cpu().numpy())
        graph_pred_bin = (graph_prediction >= 0.5)

        return graph_ground_truth, graph_pred_bin
    

    def update_metrics(self, 
                        curr_snapshot: torch.Tensor, 
                        snapshot_t: int, 
                        snapshot_idx: int,
                        metrics_list: Dict[str, List[Any]], 
                        split_mode: str):

        # Whole graph evaluation
        num_nodes = self.train_loader.dataset.num_nodes
        graph_mask = torch.ones((num_nodes, num_nodes)).to(self.device)
        ## Skip self-loop edges during evaluation
        graph_mask.fill_diagonal_(0)

        ground_truth, pred_bin = self._get_graph_mask_ground_truth_and_pred(graph_mask, curr_snapshot)
        del graph_mask
        
        precision = precision_score(ground_truth.cpu().numpy(), pred_bin)
        recall = recall_score(ground_truth.cpu().numpy(), pred_bin)
        f1 = f1_score(ground_truth.cpu().numpy(), pred_bin)
        acc = accuracy_score(ground_truth.cpu().numpy(), pred_bin)
        
        ## Record metrics in total
        metrics_list[f"avg_precision"].append(precision)
        metrics_list[f"avg_recall"].append(recall)
        metrics_list[f"avg_f1"].append(f1)
        metrics_list[f"avg_acc"].append(acc)

         ## Record metrics per snapshot_idx
        k = f"day{snapshot_idx}_avg_precision"
        if k not in metrics_list:
            metrics_list[f"day{snapshot_idx}_avg_precision"] = []
            metrics_list[f"day{snapshot_idx}_avg_recall"] = []
            metrics_list[f"day{snapshot_idx}_avg_f1"] = []
            metrics_list[f"day{snapshot_idx}_avg_acc"] = []
        metrics_list[f"day{snapshot_idx}_avg_precision"].append(precision)
        metrics_list[f"day{snapshot_idx}_avg_recall"].append(recall)
        metrics_list[f"day{snapshot_idx}_avg_f1"].append(f1)
        metrics_list[f"day{snapshot_idx}_avg_acc"].append(acc)
