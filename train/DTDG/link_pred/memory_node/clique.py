import numpy as np
import re
import os
import pickle
import torch
from typing import Dict, List, Any
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ....DTDG.link_pred.clique import LinkPredCliqueTrainer
from .....dataset.DTDG.graph_generation.mem_node import MemoryNodeGenerator

class LinkPredMemoryNodeCliqueTrainer(LinkPredCliqueTrainer):
    def __init__(self):
        super(LinkPredMemoryNodeCliqueTrainer, self).__init__()
        data_pattern = self.args.data.split("/")[0]
        match = re.fullmatch(MemoryNodeGenerator.REGEX, data_pattern)
        if match:
            self.K = int(match.group(1))

    def set_model_args(self, parser):
        return parser
        
    def update_metrics(self, 
                        curr_snapshot: torch.Tensor, 
                        snapshot_t: int, 
                        snapshot_idx: int,
                        metrics_list: Dict[str, List[Any]], 
                        split_mode: str):
        raise NotImplementedError("Empty model does not support updating metrics.")

    def list_of_metrics_names(self) -> List[str]:
        return ["memnode_avg_precision", "memnode_avg_recall", "memnode_avg_f1", "memnode_avg_acc"]

    def eval_for_one_epoch(self, split_mode: str):
        assert not torch.is_grad_enabled(), "During evaluation, torch grad should be disabled."

        if split_mode == 'train':
            eval_loader = self.train_loader
        elif split_mode == 'val':
            eval_loader = self.val_loader
        elif split_mode == 'test':
            eval_loader = self.test_loader
        elif split_mode == 'test_inductive':
            eval_loader = self.test_inductive_loader
        else:
            raise NotImplementedError()

        metrics_list = {k: [] for k in self.list_of_metrics_names()}

        for snapshot_idx, batch in enumerate(eval_loader):
            if split_mode == 'test_inductive' and snapshot_idx < self.K:
                continue

            print(f"\t\%\% Evaluation iteration {snapshot_idx} out of {len(eval_loader)}", flush=True)
            _, _, curr_snapshot, _, _, _ = batch
            assert len(curr_snapshot) == 1, "The batch size for this task should be one."

            if snapshot_idx == 0:
                prev_snapshot = eval_loader.dataset[-1][0]
                prev_snapshot = [prev_snapshot]
                
            curr_snapshot = curr_snapshot[0]
            curr_snapshot = curr_snapshot.to(self.device)
            # ground truth is the set of links between the first node and all its neighbors
            ground_truth = torch.cat([curr_snapshot[0], curr_snapshot[:, 0]])
            pred = torch.ones_like(ground_truth)

            precision = precision_score(ground_truth.flatten().long().cpu().numpy(), pred.flatten().long().cpu().numpy(), zero_division=0)
            recall = recall_score(ground_truth.flatten().long().cpu().numpy(), pred.flatten().long().cpu().numpy(), zero_division=0)
            f1 = f1_score(ground_truth.flatten().long().cpu().numpy(), pred.flatten().long().cpu().numpy(), zero_division=0)
            acc = accuracy_score(ground_truth.flatten().long().cpu().numpy(), pred.flatten().long().cpu().numpy())

            ## Record metrics in total
            metrics_list[f"memnode_avg_precision"].append(precision)
            metrics_list[f"memnode_avg_recall"].append(recall)
            metrics_list[f"memnode_avg_f1"].append(f1)
            metrics_list[f"memnode_avg_acc"].append(acc)
            
        # Take the average of all metrics among snapshots
        metrics_list = {metric: np.nan_to_num(np.mean(list_of_values), nan=-1.0) for metric, list_of_values in metrics_list.items()}
        return metrics_list 