import pickle
import re
from typing import Any, Dict, List

import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import torch

from ....CTDG.link_pred.edgebank import LinkPredEdgeBankTrainer
from .....dataset.DTDG.graph_generation.mem_node import MemoryNodeGenerator


class EdgeBankMemoryNodeTrainer(LinkPredEdgeBankTrainer):
    """ This class is only useful to train periodic-based datasets. 
    For more info about periodic datasets, please checkout the documentation of `PeriodicGenerator` class.
    """
    def __init__(self):
        super(EdgeBankMemoryNodeTrainer, self).__init__()
        data_pattern = self.args.data.split("/")[0]
        match = re.fullmatch(MemoryNodeGenerator.REGEX, data_pattern)
        if match:
            self.K = int(match.group(1))

    def get_dataset_regex_pattern(self):
        dataset_regex_pattern = MemoryNodeGenerator.REGEX
        dataset_name_part_to_check = self.args.data.split("/")[0]
        return dataset_regex_pattern, dataset_name_part_to_check

    def list_of_metrics_names(self) -> List[str]:
        return ["memnode_avg_precision", "memnode_avg_recall", "memnode_avg_f1", "memnode_avg_acc"]

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
        # The first k snapshots (K the time gap between when discovery and pattern happen) of -
        # test-inductive is not valid as being "inductive" because memory node in those steps -
        # discover patterns appeared in 'test' split which are transductive.
        # Simply skip those steps to avoid errors during evaluation.
        if split_mode == 'test_inductive' and snapshot_idx < self.K:
            return
        
        # memnode graph evaluation
        num_nodes = self.train_loader.dataset.num_nodes
        memnode_graph_mask = torch.zeros((num_nodes, num_nodes)).to(self.device)
        memnode_graph_mask[0, :] = 1
        memnode_graph_mask[:, 0] = 1

        ## Skip self-loop edges during evaluation
        memnode_graph_mask.fill_diagonal_(0)
        memnode_graph_ground_truth, memnode_graph_pred_bin = self._get_graph_mask_ground_truth_and_pred(memnode_graph_mask, curr_snapshot)
        del memnode_graph_mask
        
        memnode_graph_precision = precision_score(memnode_graph_ground_truth.cpu().numpy(), memnode_graph_pred_bin)
        memnode_graph_recall = recall_score(memnode_graph_ground_truth.cpu().numpy(), memnode_graph_pred_bin)
        memnode_graph_f1 = f1_score(memnode_graph_ground_truth.cpu().numpy(), memnode_graph_pred_bin)
        memnode_graph_acc = accuracy_score(memnode_graph_ground_truth.cpu().numpy(), memnode_graph_pred_bin)
        
        ## Record metrics in total
        metrics_list[f"memnode_avg_precision"].append(memnode_graph_precision)
        metrics_list[f"memnode_avg_recall"].append(memnode_graph_recall)
        metrics_list[f"memnode_avg_f1"].append(memnode_graph_f1)
        metrics_list[f"memnode_avg_acc"].append(memnode_graph_acc)


    def _eval_predict_current_timestep(self, curr_snapshot: torch.Tensor):
        """ This function visualizes model's output alongside ground-truth on memory-node graphs. 
        As the model's prediction on memory-node has higher importance, a subpart of graph that only shows memory-node links is visualized, and the rest is discarded.
        """
        memory_node_subgraph_mask = torch.zeros_like(curr_snapshot)
        memory_node_subgraph_mask[0] = 1
        memory_node_subgraph_mask[:, 0] = 1
        src, dst = torch.nonzero(memory_node_subgraph_mask, as_tuple=True)
        
        pred = self.model['edgebank'].predict_link(src.numpy(), dst.numpy())

        out_2d = torch.zeros_like(curr_snapshot)
        out_2d[src, dst] = torch.from_numpy(pred).float()
        out_2d.fill_diagonal_(0)

        del memory_node_subgraph_mask

        return out_2d