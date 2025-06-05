from abc import abstractmethod
import copy
import os
import re
from tqdm import tqdm
import pickle
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import to_dense_adj

from .graph_generator import GraphGenerator, nx_undirected_graph_to_sparse

import random

class SpatioTemporalLongRange(GraphGenerator):

    _pattern = r"^\((\d+), (\d+)\)$"
    REGEX = f"{_pattern}"

    def __init__(self, args) -> None:
        super(SpatioTemporalLongRange, self).__init__(args.num_nodes,
                 args.dataset_name,
                 args.seed)

        # Concat train/val/test number of weeks to the dataset name.
        self.dataset_name = self.dataset_name + f"/long_range-{args.num_samples}ns-{args.num_nodes}nn-{args.num_branches}nb-{args.val_ratio}vr-{args.test_ratio}tr"
        self.args = args

        lag_dist = self.dataset_name.split("/")[0]
        if re.fullmatch(SpatioTemporalLongRange._pattern, lag_dist):
            match = re.fullmatch(SpatioTemporalLongRange._pattern, lag_dist)
            self.lag = int(match.group(1))
            self.branch_len = int(match.group(2))
        else:
            raise NotImplementedError()

        self.T = self.args.num_samples + self.lag

        # Indexing on nodes during training requires 64-bit integers.
        # src and dst nodes data types.
        self.node_datatype = np.int64

        # time data type
        if self.T < 2**8:
            t_datatype = np.int16
        elif self.T < 2**16:
            t_datatype = np.int32
        else:
            t_datatype = np.int64
        
        self.t_datatype = t_datatype

        self.EDGE_FEAT=1
        # edge feature data type
        if self.EDGE_FEAT < 2**8:
            edge_feat_datatype = np.int16
        elif self.EDGE_FEAT < 2**16:
            edge_feat_datatype = np.int32
        else:
            edge_feat_datatype = np.int64

        self.edge_feat_datatype = edge_feat_datatype

        # Specify the number of nodes for the discovery, and inductive/transductive patterns.
        self.effect_node = 0 # Node zero is the discovery node.
        self.cause_node = 1

        # Specify the start time of different data splits.
        self.test_num_samples = int(self.T * self.args.test_ratio)
        self.val_num_samples = int(self.T * self.args.val_ratio)
        self.train_num_samples = self.T - self.test_num_samples - self.val_num_samples
        
        self.train_start_t = 0
        self.val_start_t = self.train_start_t + self.train_num_samples
        self.test_start_t = self.val_start_t + self.val_num_samples

        assert self.T == self.test_start_t + self.test_num_samples, f"Total time should be equal to the sum of test 'time' and 'number of samples'. Got {self.T} and {self.test_start_t + self.test_num_samples} instead."

    @staticmethod
    def get_parser():
        parser = GraphGenerator.get_parser()
        parser.add_argument("--val-ratio", type=float, required=True)
        parser.add_argument("--test-ratio", type=float, required=True)
        parser.add_argument("--visualize", action="store_true")
        parser.add_argument("--num-samples", type=int, required=True)

        parser.add_argument("--num-branches", type=int, required=True)

        return parser

    def get_val_mask(self, t: np.ndarray) -> torch.Tensor:
        return np.logical_and(t >= self.val_start_t, t < self.test_start_t)

    def get_test_mask(self, t: np.ndarray) -> torch.Tensor:
        return t >= self.test_start_t
    
    def get_test_inductive_mask(self, t: np.ndarray) -> torch.Tensor:
        return np.zeros_like(t, dtype=bool)
        
    def _create_star_with_paths(self):
        G = nx.Graph()
        G.add_node(self.effect_node)
        G.add_node(self.cause_node)
        center_node = self.cause_node

        current_node = center_node + 1  # Start labeling from next node ID
        end_nodes = []
        
        for i in range(self.args.num_branches):
            prev_node = center_node
            for j in range(self.branch_len):
                G.add_node(current_node)
                G.add_edge(prev_node, current_node)
                prev_node = current_node
                end_node = current_node
                current_node += 1
            end_nodes.append(end_node)
        
        for i in range(current_node, self.args.num_nodes):
            G.add_node(current_node)
            current_node += 1
        
        assert G.number_of_nodes() == self.num_nodes, f"Number of nodes should be {self.num_nodes}. Got {G.number_of_nodes()} instead."

        return G, end_nodes

    def reorder_nodes_wo_cause_effect(self, num_nodes: int):
        # Step 1: Generate a permutation for indices 2 to num_nodes - 1
        old_indices = np.arange(num_nodes)
        old_indices_wo_cause_effect = old_indices[2:]
        new_indices_wo_cause_effect = np.random.permutation(old_indices_wo_cause_effect)
        new_indices = np.concatenate(([self.effect_node, self.cause_node], new_indices_wo_cause_effect))

        return new_indices
    

    def get_links(self, args_dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        now_t = 0

        src = np.empty(0, dtype=self.node_datatype)
        dst = np.empty_like(src)
        t = np.empty_like(src, dtype=self.t_datatype)
        edge_feat = np.empty_like(src, dtype=self.edge_feat_datatype)

        def _update_sparse_data(G: nx.Graph):
            nonlocal src, dst, t, edge_feat
            src_t, dst_t, edge_feat_t = nx_undirected_graph_to_sparse(G, return_edge_feat=True)
            src = np.concatenate([src, src_t])
            dst = np.concatenate([dst, dst_t])
            t = np.concatenate([t, np.full_like(src_t, fill_value=now_t)])
            edge_feat = np.concatenate([edge_feat, edge_feat_t])
        
        pos = nx.circular_layout(nx.complete_graph(self.num_nodes))

        def _vis_graph(G, idx, stage):
            nonlocal pos
            plt.figure(figsize=(20, 10))
            nx.draw_networkx(G, pos, node_size=40, with_labels=True, node_color="yellow")
            # Draw edge labels for the 'weight' attribute
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

            plt.title(f"Day {idx} out of {self.T}.")
            vis_save_dir = os.path.join(args_dict["save_dir"], self.dataset_name, "vis", stage)
            os.makedirs(vis_save_dir, exist_ok=True)
            plt.savefig(os.path.join(vis_save_dir, f"{idx}.png"))
            plt.close()
        
        G_star, end_nodes = self._create_star_with_paths()
        original_node_ids = np.arange(G_star.number_of_nodes())
        new_node_ids = original_node_ids
        with tqdm(total=self.T, desc=self.dataset_name) as pbar:
            graph_patterns: List[List[int]] = []
            # Generate graph between t=0,...,lag-1
            for i in range(self.lag):
                G: nx.Graph = nx.empty_graph(G_star.number_of_nodes())
                G.add_edges_from(G_star.edges(data=True))
                G = nx.relabel_nodes(G, dict(zip(original_node_ids, new_node_ids)))
                graph_patterns.append(new_node_ids[end_nodes])

                _update_sparse_data(G)

                if self.args.visualize:
                    _vis_graph(G, i, stage="t=0,...,k-1")

                new_node_ids = self.reorder_nodes_wo_cause_effect(G_star.number_of_nodes())

                now_t += 1
                pbar.update(1)

            # Generate graph between t=lag,...,test
            for i in range(self.lag, self.T):
                G: nx.Graph = nx.empty_graph(G_star.number_of_nodes())
                G.add_edges_from(G_star.edges(data=True))
                G = nx.relabel_nodes(G, dict(zip(original_node_ids, new_node_ids)))
                target_end_nodes = graph_patterns.pop(0)
                G.add_edges_from([(self.effect_node, i) for i in target_end_nodes])
                graph_patterns.append(new_node_ids[end_nodes])

                _update_sparse_data(G)


                if self.args.visualize:
                    if i < 4 * self.lag:
                        _vis_graph(G, i, stage="t=k,...,inductive_test_t")

                new_node_ids = self.reorder_nodes_wo_cause_effect(G_star.number_of_nodes())

                now_t += 1
                pbar.update(1)

        assert t.max() == self.T - 1, f"Last timestep should be {self.T - 1}. Got {t.max()} instead."

        return src.astype(self.node_datatype), dst.astype(self.node_datatype), t.astype(self.t_datatype), edge_feat.astype(self.edge_feat_datatype)