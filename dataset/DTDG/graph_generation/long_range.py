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

class LongRange(GraphGenerator):
    """ This class implements Long-Range Temporal Models. The dataset has multiple snapshots of a graph, each representing random patterns and long-appearance discovery patterns. 
    The dataset name generally looks like this: (K, N, L)/[specific name], where K represents the gap between random patterns, N represents the number of random patterns, 
    and L represents the duration of pattern appearances.
    """

    _pattern = r"^\((\d+), (\d+)\)$"
    REGEX = f"{_pattern}"

    def __init__(self, args) -> None:
        super(LongRange, self).__init__(args.num_nodes,
                 args.dataset_name,
                 args.neg_sampling_strategy, 
                 args.seed, 
                 args.num_neg_links_to_sample_per_pos_link,
                 args.do_neg_sampling)

        # Concat train/val/test number of weeks to the dataset name.
        self.dataset_name = self.dataset_name + f"/long_range-{args.num_nodes}n-{args.pattern_mode}pm-{args.val_ratio}vr-{args.test_ratio}tr"

        if args.pattern_mode == "er":
            self.dataset_name = self.dataset_name + f"-{args.er_pattern_extra}eps-{args.er_prob_pattern}epp"
        else:
            raise NotImplementedError()

        self.args = args

        long_range_frequency = self.dataset_name.split("/")[0]
        if re.fullmatch(LongRange._pattern, long_range_frequency):
            match = re.fullmatch(LongRange._pattern, long_range_frequency)
            self.long_range_freq = int(match.group(1))
            self.repetition = int(match.group(2))
        else:
            raise NotImplementedError()

        self.T = self.repetition * self.long_range_freq

        # src and dst nodes data types.
        if self.num_nodes < 2**8:
            node_datatype = np.uint8
        elif self.num_nodes < 2**16:
            node_datatype = np.uint16
        elif self.num_nodes < 2**32:
            node_datatype = np.uint32
        else:
            node_datatype = np.uint64
        self.node_datatype = node_datatype

        # time data type
        if self.T < 2**8:
            t_datatype = np.uint8
        elif self.T < 2**16:
            t_datatype = np.uint16
        elif self.T < 2**32:
            t_datatype = np.uint32
        else:
            t_datatype = np.uint64
        self.t_datatype = t_datatype

        self.EDGE_FEAT=1
        # edge feature data type
        if self.EDGE_FEAT < 2**8:
            edge_feat_datatype = np.uint8
        elif self.EDGE_FEAT < 2**16:
            edge_feat_datatype = np.uint16
        elif self.EDGE_FEAT < 2**32:
            edge_feat_datatype = np.uint32
        else:
            edge_feat_datatype = np.uint64
        self.edge_feat_datatype = edge_feat_datatype

        # Specify the number of nodes for the discovery, and inductive/transductive patterns.
        self.discovery_num_nodes = 1 # Node zero is the discovery node.
        self.pattern_num_nodes = self.num_nodes - self.discovery_num_nodes

        # Specify the start time of different data splits.
        self.test_start_t = self.T - int(self.T * self.args.test_ratio)
        self.val_start_t = self.test_start_t - int(self.T * self.args.val_ratio)
        self.train_start_t = 0

    @staticmethod
    def get_parser():
        parser = GraphGenerator.get_parser()
        parser.add_argument("--pattern-mode", type=str, choices=['er'], required=True)
        parser.add_argument("--val-ratio", type=float, required=True)
        parser.add_argument("--test-ratio", type=float, required=True)
        parser.add_argument("--visualize", action="store_true")

        # arguments specific for ER pattern
        # er-prob-pattern should be much greater than er-prob-extra
        parser.add_argument("--er-prob-pattern", type=float, required=True)
        parser.add_argument("--er-pattern-extra", type=float, required=True, help="Duration of pattern appearances.")

        return parser

    def get_val_mask(self, t: np.ndarray) -> torch.Tensor:
        return np.logical_and(t >= self.val_start_t, t < self.test_start_t)

    def get_test_mask(self, t: np.ndarray) -> torch.Tensor:
        return t >= self.test_start_t
    
    def generate_pattern_graph(self, G_pattern: nx.Graph, p: float) -> None:
        if self.args.pattern_mode == "er":
            # p = self.args.er_prob
            er_graph = nx.erdos_renyi_graph(n=G_pattern.number_of_nodes(), p=p)
            G_pattern.add_edges_from(er_graph.edges)
        else:
            raise NotImplementedError()

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
            nx.draw_networkx(G, pos, node_size=40, with_labels=True, node_color="yellow")
            plt.title(f"Day {idx} out of {self.T}.")
            vis_save_dir = os.path.join(GraphGenerator.SAVE_DIR, self.dataset_name, "vis", stage)
            os.makedirs(vis_save_dir, exist_ok=True)
            plt.savefig(os.path.join(vis_save_dir, f"{idx}.png"))
            plt.close()

        with tqdm(total=self.T, desc=self.dataset_name) as pbar:
            graph_patterns: List[nx.Graph] = []

            # If fixed a graph pattern for the pick parts, 0, 100, 200, etc
            G_0: nx.Graph = nx.empty_graph(self.num_nodes)
            G_pattern: nx.Graph = nx.empty_graph(n=self.num_nodes)
            self.generate_pattern_graph(G_pattern, args_dict["er_prob_pattern"])        
            G_0.add_edges_from(G_pattern.edges)

            
            for i in range(self.T):
                G: nx.Graph = nx.empty_graph(self.num_nodes)
                if i % self.long_range_freq == 0:
                    G.add_edges_from(G_0.edges)
                else:
                    G_pattern: nx.Graph = nx.empty_graph(n=self.num_nodes)
                    self.generate_pattern_graph(G_pattern, args_dict["er_pattern_extra"])
                    G.add_edges_from(G_pattern.edges)

                _update_sparse_data(G)
                if self.args.visualize and i < self.long_range_freq + 2:
                    _vis_graph(G, i, stage="all")

                now_t += 1
                pbar.update(1)

        return src, dst, t, edge_feat
