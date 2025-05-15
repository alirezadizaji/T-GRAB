from typing import Any, Dict, List
from argparse import ArgumentParser
import os
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import timeit
import torch
from tgb.utils.utils import set_random_seed, save_results

from .....utils import  EarlyStopMonitor
from ....CTDG.link_pred.trainer import LinkPredTrainer
from .....dataset.DTDG.graph_generation.periodicity import PeriodicGenerator

class PeriodicTrainer(LinkPredTrainer):
    """ This class is only useful to train periodic-based datasets. 
    For more info about periodic datasets, please checkout the documentation of `PeriodicGenerator` class.
    """

    def get_dataset_regex_pattern(self):
        dataset_regex_pattern = PeriodicGenerator.PERIODIC_REGEX
        dataset_name_part_to_check = self.args.data.split("/")[0]
        return dataset_regex_pattern, dataset_name_part_to_check
   
    def list_of_metrics_names(self) -> List[str]:
        return ["avg_precision", "avg_recall", "avg_f1", "avg_acc"]
        
    def _get_graph_mask_ground_truth_and_pred(self, graph_mask, dataset, curr_snapshot, snapshot_t):
        # Compute precision, recall, f1, and accuracy on the whole graph snapshot
        graph_src_indices, graph_dst_indices = torch.nonzero(graph_mask, as_tuple=True)
        graph_ground_truth = curr_snapshot[graph_src_indices, graph_dst_indices].float()
        graph_pos_mask = graph_ground_truth > 0
        graph_neg_mask = ~graph_pos_mask
        graph_pos_src_indices, graph_pos_dst_indices = \
                graph_src_indices[graph_pos_mask], graph_dst_indices[graph_pos_mask]
        graph_neg_src_indices, graph_neg_dst_indices = \
                graph_src_indices[graph_neg_mask], graph_dst_indices[graph_neg_mask]
        graph_pos_t = torch.full_like(graph_pos_src_indices, fill_value=snapshot_t)
        graph_neg_t = torch.full_like(graph_neg_src_indices, fill_value=snapshot_t)
        graph_pos_edge_ids, graph_pos_edge_feats = dataset.get_attr(
                                                            graph_pos_src_indices, 
                                                            graph_pos_dst_indices, 
                                                            graph_pos_t, attrs=["edge_ids", "edge_feat"])
        
        (graph_pos_src_nodes_embeddings, graph_pos_dst_nodes_embeddings), (graph_neg_src_nodes_embeddings, graph_neg_dst_nodes_embeddings) = \
                            self.forward_backbone(
                                graph_pos_src_indices, 
                                graph_pos_dst_indices, 
                                graph_pos_t,
                                batch_edge_id=graph_pos_edge_ids,
                                batch_edge_feat=graph_pos_edge_feats,
                                batch_neg=(graph_neg_src_indices, graph_neg_dst_indices, graph_neg_t))    

        embedding_dims = graph_pos_src_nodes_embeddings.shape[1:]
        
        graph_src_nodes_embeddings = torch.empty((graph_src_indices.numel(), *embedding_dims), device=self.device)
        graph_src_nodes_embeddings.fill_(torch.nan)
        graph_dst_nodes_embeddings = torch.empty_like(graph_src_nodes_embeddings)
        graph_dst_nodes_embeddings.fill_(torch.nan)
        
        graph_src_nodes_embeddings[graph_pos_mask] = graph_pos_src_nodes_embeddings
        graph_dst_nodes_embeddings[graph_pos_mask] = graph_pos_dst_nodes_embeddings
        graph_src_nodes_embeddings[graph_neg_mask] = graph_neg_src_nodes_embeddings
        graph_dst_nodes_embeddings[graph_neg_mask] = graph_neg_dst_nodes_embeddings

        # Make sure the embeddings for both src and dst nodes are found.
        assert torch.all(torch.isfinite(graph_src_nodes_embeddings))
        assert torch.all(torch.isfinite(graph_dst_nodes_embeddings))
        
        graph_prediction = self.model['link_pred'](graph_src_nodes_embeddings, graph_dst_nodes_embeddings)
        graph_pred_bin = (graph_prediction >= 0.5).squeeze(-1)

        return graph_ground_truth, graph_pred_bin

    def update_metrics(self, 
                        curr_snapshot: torch.Tensor, 
                        snapshot_t: int, 
                        snapshot_idx: int,
                        metrics_list: Dict[str, List[Any]], 
                        split_mode: str,
                        dataset):
                        
        # Whole graph evaluation
        num_nodes = self.train_loader.dataset.num_nodes
        graph_mask = torch.ones((num_nodes, num_nodes)).to(self.device)
        graph_mask.fill_diagonal_(0)

        ground_truth, pred_bin = self._get_graph_mask_ground_truth_and_pred(graph_mask, 
                                                                            dataset, 
                                                                            curr_snapshot, 
                                                                            snapshot_t)
        del graph_mask
        
        precision = precision_score(ground_truth.cpu().numpy(), pred_bin.cpu().numpy(), zero_division=0)
        recall = recall_score(ground_truth.cpu().numpy(), pred_bin.cpu().numpy(), zero_division=0)
        f1 = f1_score(ground_truth.cpu().numpy(), pred_bin.cpu().numpy(), zero_division=0)
        acc = accuracy_score(ground_truth.cpu().numpy(), pred_bin.cpu().numpy())
        
        ## Record metrics in total
        metrics_list[f"avg_precision"].append(precision)
        metrics_list[f"avg_recall"].append(recall)
        metrics_list[f"avg_f1"].append(f1)
        metrics_list[f"avg_acc"].append(acc)

        ## Record metrics per snapshot_idx
        if split_mode != "train":
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


    def early_stopping_checker(self, early_stopper) -> bool:
        if self.test_perf[self.val_first_metric] == 1:
            return True
        
        return super().early_stopping_checker(early_stopper)