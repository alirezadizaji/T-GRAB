from typing import Any, Dict, List
import os
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import timeit
import torch
from tgb.utils.utils import set_random_seed, save_results

from .....utils import  EarlyStopMonitor
from ....CTDG.link_pred.trainer import LinkPredTrainer
from .....dataset.DTDG.graph_generation.long_range import LongRange

class LongRangeTrainer(LinkPredTrainer):
    """ This class is only useful to train long-range-based datasets. 
    For more info about long-range datasets, please checkout the documentation of `LongRange` class.
    """
    
    def __init__(self):
        super(LongRangeTrainer, self).__init__()
        data_pattern = self.args.data.split("/")[0]
        match = re.fullmatch(LongRange.REGEX, data_pattern)
        if match:
            self.K = int(match.group(1))
    
    def get_dataset_regex_pattern(self):
        dataset_regex_pattern = LongRange.REGEX
        dataset_name_part_to_check = self.args.data.split("/")[0]
        return dataset_regex_pattern, dataset_name_part_to_check

    def list_of_metrics_names(self) -> List[str]:
        return [f"day{day}_avg_precision" for day in range(self.K)] + \
               [f"day{day}_avg_recall" for day in range(self.K)] + \
               [f"day{day}_avg_f1" for day in range(self.K)] + \
               [f"day{day}_avg_acc" for day in range(self.K)] + \
               ["avg_precision", "avg_recall", "avg_f1", "avg_acc"]

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
        
        # Long range dependency happens periodically, while in a period there is only one (which is first in period) - 
        # deterministic pattern that is important.
        # During Training, This can help reduce evaluation overhead time significantly.
        if (not self.args.eval_mode) and (snapshot_idx % self.K != 0):
            return
                 
        # Whole graph evaluation
        num_nodes = self.train_loader.dataset.num_nodes
        whole_graph_mask = torch.ones((num_nodes, num_nodes)).to(self.device)
        ## Skip self-loop edges during evaluation
        whole_graph_mask.fill_diagonal_(0)
        whole_graph_ground_truth, whole_graph_pred_bin = self._get_graph_mask_ground_truth_and_pred(whole_graph_mask, 
                                                                                                    dataset, 
                                                                                                    curr_snapshot, 
                                                                                                    snapshot_t)
        del whole_graph_mask
        
        whole_graph_precision = precision_score(whole_graph_ground_truth.cpu().numpy(), whole_graph_pred_bin.cpu().numpy(), zero_division=0)
        whole_graph_recall = recall_score(whole_graph_ground_truth.cpu().numpy(), whole_graph_pred_bin.cpu().numpy(), zero_division=0)
        whole_graph_f1 = f1_score(whole_graph_ground_truth.cpu().numpy(), whole_graph_pred_bin.cpu().numpy(), zero_division=0)
        whole_graph_acc = accuracy_score(whole_graph_ground_truth.cpu().numpy(), whole_graph_pred_bin.cpu().numpy())
        
        ## Record metrics in total
        metrics_list[f"avg_precision"].append(whole_graph_precision)
        metrics_list[f"avg_recall"].append(whole_graph_recall)
        metrics_list[f"avg_f1"].append(whole_graph_f1)
        metrics_list[f"avg_acc"].append(whole_graph_acc)

        ## Record metrics per day
        day = snapshot_idx % self.K
        metrics_list[f"day{day}_avg_precision"].append(whole_graph_precision)
        metrics_list[f"day{day}_avg_recall"].append(whole_graph_recall)
        metrics_list[f"day{day}_avg_f1"].append(whole_graph_f1)
        metrics_list[f"day{day}_avg_acc"].append(whole_graph_acc)