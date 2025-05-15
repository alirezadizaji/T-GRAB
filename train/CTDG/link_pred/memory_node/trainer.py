from typing import Any, Dict, List
from argparse import ArgumentParser
import os
import re
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import timeit
import torch
from torch.utils.data import Dataset
from tgb.utils.utils import set_random_seed, save_results

from .....utils import  EarlyStopMonitor
from ....CTDG.link_pred.trainer import LinkPredTrainer
from .....dataset.DTDG.graph_generation.mem_node import MemoryNodeGenerator
from ....CTDG.trainer import CTDGTrainer, NODE_EMB_MODEL_NAME


class MemoryNodeTrainer(LinkPredTrainer):
    """ This class is only useful to train periodic-based datasets. 
    For more info about periodic datasets, please checkout the documentation of `PeriodicGenerator` class.
    """
    def __init__(self):
        super(MemoryNodeTrainer, self).__init__()
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
        
    def _get_graph_mask_ground_truth_and_pred(self, graph_mask, dataset, curr_snapshot, snapshot_t):
        # Compute precision, recall, f1, and accuracy on the memnode graph snapshot
        graph_src_indices, graph_dst_indices = torch.nonzero(graph_mask, as_tuple=True)
        graph_ground_truth = curr_snapshot[graph_src_indices, graph_dst_indices].float()
        graph_pos_mask = graph_ground_truth > 0
        graph_neg_mask = ~graph_pos_mask
        graph_pos_src_indices, graph_pos_dst_indices = \
                graph_src_indices[graph_pos_mask], graph_dst_indices[graph_pos_mask]
        graph_neg_src_indices, graph_neg_dst_indices = \
                graph_src_indices[graph_neg_mask], graph_dst_indices[graph_neg_mask]
        graph_pos_t = torch.full_like(graph_pos_src_indices, fill_value=snapshot_t)
        graph_pos_edge_ids, graph_pos_edge_feats = dataset.get_attr(
                                                            graph_pos_src_indices, 
                                                            graph_pos_dst_indices, 
                                                            graph_pos_t, attrs=["edge_ids", "edge_feat"])
        graph_neg_t = torch.full_like(graph_neg_src_indices, fill_value=snapshot_t)
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
        # The first k snapshots (K the time gap between when discovery and pattern happen) of -
        # test-inductive is not valid as being "really" inductive, because memory node in those steps -
        # discover patterns appeared in 'test' split which those patterns are transductive.
        # Simply skip those steps to avoid wrong interpretations during evaluation.
        if split_mode == 'test_inductive' and snapshot_idx < self.K:
            return
        
        # Memory node graph evaluation
        num_nodes = self.train_loader.dataset.num_nodes
        memnode_graph_mask = torch.zeros((num_nodes, num_nodes)).to(self.device)
        memnode_graph_mask[0, :] = 1
        memnode_graph_mask[:, 0] = 1
        ## Skip self-loop edges during evaluation
        memnode_graph_mask.fill_diagonal_(0)

        # Skip evaluation if memory node is inactive
        if not torch.any(memnode_graph_mask[curr_snapshot]):
            return
        
        memnode_graph_ground_truth, memnode_graph_pred_bin = self._get_graph_mask_ground_truth_and_pred(memnode_graph_mask, 
                                                                                                    dataset, 
                                                                                                    curr_snapshot, 
                                                                                                    snapshot_t)
        del memnode_graph_mask
        
        memnode_graph_precision = precision_score(memnode_graph_ground_truth.cpu().numpy(), memnode_graph_pred_bin.cpu().numpy(), zero_division=0)
        memnode_graph_recall = recall_score(memnode_graph_ground_truth.cpu().numpy(), memnode_graph_pred_bin.cpu().numpy(), zero_division=0)
        memnode_graph_f1 = f1_score(memnode_graph_ground_truth.cpu().numpy(), memnode_graph_pred_bin.cpu().numpy(), zero_division=0)
        memnode_graph_acc = accuracy_score(memnode_graph_ground_truth.cpu().numpy(), memnode_graph_pred_bin.cpu().numpy())
        
        ## Record metrics in total
        metrics_list[f"memnode_avg_precision"].append(memnode_graph_precision)
        metrics_list[f"memnode_avg_recall"].append(memnode_graph_recall)
        metrics_list[f"memnode_avg_f1"].append(memnode_graph_f1)
        metrics_list[f"memnode_avg_acc"].append(memnode_graph_acc)

    def _eval_predict_current_timestep(self, curr_snapshot: torch.Tensor, snapshot_t: int, dataset: Dataset):
        """ This function visualizes model's output alongside ground-truth on memory-node graphs. 
        As the model's prediction on memory-node has higher importance, a subpart of graph that only shows memory-node links is visualized, and the rest is discarded.
        """
        
        memory_node_subgraph_mask = torch.zeros_like(curr_snapshot)
        memory_node_subgraph_mask[0] = 1
        memory_node_subgraph_mask[:, 0] = 1
        src_subgraph_mask, dst_subgraph_mask = torch.nonzero(memory_node_subgraph_mask, as_tuple=True)
        t_subgraph_mask = torch.full_like(src_subgraph_mask, fill_value=snapshot_t)

        ground_truth = curr_snapshot[src_subgraph_mask, dst_subgraph_mask]
        pos = ground_truth > 0
        neg = ~pos

        pos_edge_ids, pos_edge_feats = dataset.get_attr(src_subgraph_mask[pos], 
                                                        dst_subgraph_mask[pos], 
                                                        t_subgraph_mask[pos], attrs=["edge_ids", "edge_feat"])

        (pos_src_node_embeddings, pos_dst_node_embeddings), (neg_src_node_embeddings, neg_dst_node_embeddings) = \
                        self.forward_backbone(src_subgraph_mask[pos], 
                                            dst_subgraph_mask[pos], 
                                            t_subgraph_mask[pos],
                                            batch_edge_id=pos_edge_ids,
                                            batch_edge_feat=pos_edge_feats,
                                            batch_neg=(src_subgraph_mask[neg], dst_subgraph_mask[neg], t_subgraph_mask[neg]))

        pos_pred = self.model['link_pred'](pos_src_node_embeddings, pos_dst_node_embeddings)
        neg_pred = self.model['link_pred'](neg_src_node_embeddings, neg_dst_node_embeddings)
        
        out_2d = torch.zeros_like(curr_snapshot).cpu()
        out_2d[src_subgraph_mask[pos], dst_subgraph_mask[pos]] = pos_pred.squeeze(-1).detach().cpu()
        out_2d[src_subgraph_mask[neg], dst_subgraph_mask[neg]] = neg_pred.squeeze(-1).detach().cpu()
        out_2d.fill_diagonal_(0)

        del memory_node_subgraph_mask

        return out_2d


    def early_stopping_checker(self, early_stopper) -> bool:
        if self.test_perf[self.val_first_metric] == 1:
            return True
        
        return super().early_stopping_checker(early_stopper)

    def _train_for_one_epoch_snapshot_based(self):
        self.before_epoch_training()

        self.model[NODE_EMB_MODEL_NAME].train()
        self.model['link_pred'].train()
        train_losses = []
        train_memnode_losses = []
        train_non_memnode_losses = []

        # Each batch represents only one snapshot.
        num_batches = len(self.train_loader)
        num_batches_to_log = 4
        batches_chunk = num_batches // num_batches_to_log
        
        for batch_idx, batch in enumerate(self.train_loader):
            if batch_idx % batches_chunk == 0:
                print(f"\t\%\% Training iteration {batch_idx} out of {num_batches}", flush=True)
            self.optim.zero_grad()
            
            # `mask` is created within the collate function.
            # `mask` enables to pass multiple snapshots in one batch, as it is highly possible that each one can have different number of real links.
            (pos_src, pos_dst), _, curr_t, pos_edge_feat, pos_edge_ids, mask = batch
            seq_len = pos_src.shape[1]
            curr_t = curr_t.unsqueeze(1).repeat(1, seq_len)
            pos_src = pos_src[mask].to(self.device)
            pos_dst = pos_dst[mask].to(self.device)
            pos_edge_feat = pos_edge_feat[mask].to(self.device)
            pos_edge_ids = pos_edge_ids[mask].to(self.device)
            pos_t = curr_t[mask].to(self.device)
            
            neg_src, neg_dst = self.get_neg_link(pos_src, pos_dst)
            neg_t = pos_t.clone()
            assert pos_src.size() == neg_src.size(), f"Number of negative links ({neg_src.size()}) is not the same as positive ones ({pos_src.size()}."
            # assert curr_t.unique().numel() == 1, "All timestamps should be the same."
            
            # Forward model that is behind an MLP model to encode node embeddings. 
            (pos_src_node_embeddings, pos_dst_node_embeddings), (neg_src_node_embeddings, neg_dst_node_embeddings) = \
                    self.forward_backbone(pos_src, 
                                        pos_dst, 
                                        pos_t,
                                        batch_edge_id=pos_edge_ids, 
                                        batch_edge_feat=pos_edge_feat,
                                        batch_neg=(neg_src, neg_dst, neg_t))
            
            assert pos_src_node_embeddings.shape[0] == pos_src.shape[0], f"Mistmatch size between backbone output ({pos_src_node_embeddings.size()}) and input ({pos_src.size()})."
            assert pos_src_node_embeddings.shape[0] == pos_src.shape[0], f"Mistmatch size between backbone output ({pos_src_node_embeddings.size()}) and input ({pos_src.size()})."

            # forward link prediction model
            pos_pred: torch.Tensor = self.model['link_pred'](pos_src_node_embeddings, pos_dst_node_embeddings)
            neg_pred: torch.Tensor = self.model['link_pred'](neg_src_node_embeddings, neg_dst_node_embeddings)

            assert torch.all(pos_pred <= 1) and torch.all(pos_pred >= 0), "Make sure predictions are in range [0, 1]."
            assert pos_src_node_embeddings.shape[0] == pos_src.shape[0], f"Mistmatch size between backbone output ({pos_src_node_embeddings.size()}) and input ({pos_src.size()})."

            pos_memnode_mask = torch.logical_or(pos_src == 0, pos_dst == 0)
            neg_memnode_mask = torch.logical_or(neg_src == 0, neg_dst == 0)
            pos_non_memnode_mask = ~pos_memnode_mask
            neg_non_memnode_mask = ~neg_memnode_mask

            pos_memnode_pred = pos_pred[pos_memnode_mask]
            neg_memnode_pred = neg_pred[neg_memnode_mask]
            pos_non_memnode_pred = pos_pred[pos_non_memnode_mask]
            neg_non_memnode_pred = neg_pred[neg_non_memnode_mask]

            # Loss computation
            loss_memnode = self.criterion(pos_memnode_pred, torch.ones_like(pos_memnode_pred)) * pos_memnode_pred.numel() / pos_pred.numel()
            loss_memnode = loss_memnode + self.criterion(neg_memnode_pred, torch.zeros_like(neg_memnode_pred)) * neg_memnode_pred.numel() / neg_pred.numel()
            
            loss_non_memnode = self.criterion(pos_non_memnode_pred, torch.ones_like(pos_non_memnode_pred)) * pos_non_memnode_pred.numel() / pos_pred.numel()
            loss_non_memnode = loss_non_memnode + self.criterion(neg_non_memnode_pred, torch.zeros_like(neg_non_memnode_pred)) * neg_non_memnode_pred.numel() / neg_pred.numel()
            
            loss = 0.0

            if not torch.isnan(loss_non_memnode):
                loss = loss + loss_non_memnode

            # First few snapshots does not have any positive links on the memory node.
            # So, we need to skip those snapshots to avoid NaN loss.
            if not torch.isnan(loss_memnode):
                loss = loss + loss_memnode

            loss.backward()
            self.optim.step()
            
            train_losses.append(loss.detach().item())
            if not torch.isnan(loss_memnode):
                train_memnode_losses.append(loss_memnode.detach().item())
            if not torch.isnan(loss_non_memnode):
                train_non_memnode_losses.append(loss_non_memnode.detach().item())

            self.after_iteration_training()

        self.after_epoch_training()

        avg_loss = np.mean(train_losses)
        avg_memnode_loss = np.mean(train_memnode_losses)
        avg_non_memnode_loss = np.mean(train_non_memnode_losses)
        
        print(f"Epoch: {self.epoch:02d}, Loss: {avg_loss:.4f}, Memnode Loss: {avg_memnode_loss:.4f}, Non-memnode Loss: {avg_non_memnode_loss:.4f}.")
        
        perf_metrics = {
            "loss": avg_loss,
            "memnode_loss": avg_memnode_loss,
            "non_memnode_loss": avg_non_memnode_loss,
        }
       
        return perf_metrics
