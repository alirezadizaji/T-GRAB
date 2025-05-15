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


class MemoryNodeGenerator(GraphGenerator):
    """ This class implements Memory node dataset. The dataset has multiple snapshots of a graph, each representing a random pattern + a discovery pattern.
    The name of dataset generally looks like this: (K, N)/[specific name] where K represents the gap between the random pattern 
    and the discovery, N represents number of random patterns to generate in a time-series sequence.
    """
    
    _pattern = r"^\((\d+), (\d+)\)$"
    REGEX = f"{_pattern}"

    def __init__(self, args) -> None:
        super(MemoryNodeGenerator, self).__init__(args.num_nodes,
                 args.dataset_name,
                 args.neg_sampling_strategy, 
                 args.seed, 
                 args.num_neg_links_to_sample_per_pos_link,
                 args.do_neg_sampling)

        # Concat train/val/test number of weeks to the dataset name.
        self.dataset_name = self.dataset_name + f"/memory_node{args.t_unit}t-{args.num_nodes}n-{args.pattern_mode}pm-{args.val_ratio}vr-{args.test_ratio}tr-{args.test_inductive_ratio}tir-{args.test_inductive_num_nodes_ratio}tinnr"

        if args.pattern_mode in ["er", "nonlinear_er", "nonlin_er_const_w_had", "nonlin_er_const_w_dot", "nonlin_er_a2_2nd_hop"]:
            self.dataset_name = self.dataset_name + f"-{args.er_prob}ep-{args.er_prob_inductive}epi"
        if args.pattern_mode in ["nonlinear_er", "nonlin_er_const_w_had", "nonlin_er_const_w_dot"]:
            self.dataset_name = self.dataset_name + f"-{args.mean}mean-{args.std}std"
        if args.pattern_mode in ["nonlin_er_const_w_had", "nonlin_er_const_w_dot"]:
            self.dataset_name = self.dataset_name + f"-{args.weight_seed}wseed"
        else:
            NotImplementedError()
        
        self.args = args

        num_of_gap_and_patterns = self.dataset_name.split("/")[0]
        if re.fullmatch(MemoryNodeGenerator._pattern, num_of_gap_and_patterns):
            match = re.fullmatch(MemoryNodeGenerator._pattern, num_of_gap_and_patterns)
            self.gap = int(match.group(1))
            self.num_patterns = int(match.group(2))
        else:
            raise NotImplementedError()
        
        self.total_num_samples = self.gap + self.num_patterns
        self.T = self.total_num_samples * self.args.t_unit
        
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
        self.discovery_num_nodes = 1 # Node zero is the discovery node.
        self.pattern_num_nodes = self.num_nodes - self.discovery_num_nodes
        self.pattern_num_nodes_inductive = int(self.pattern_num_nodes * self.args.test_inductive_num_nodes_ratio)
        self.pattern_num_nodes_transductive = self.pattern_num_nodes - self.pattern_num_nodes_inductive
        self.pattern_1st_node_idx_transductive = self.discovery_num_nodes
        self.pattern_1st_node_idx_inductive = self.pattern_1st_node_idx_transductive + self.pattern_num_nodes_transductive

        # Specify the start time of different data splits.
        self.test_inductive_num_samples = int(self.total_num_samples * self.args.test_inductive_ratio)
        self.test_transductive_num_samples = int(self.total_num_samples * self.args.test_ratio)
        self.val_num_samples = int(self.total_num_samples * self.args.val_ratio)
        self.train_num_samples = self.total_num_samples - self.test_inductive_num_samples - self.test_transductive_num_samples - self.val_num_samples
        
        self.train_start_t = 0
        self.val_start_t = self.train_start_t + self.train_num_samples * self.args.t_unit
        self.test_transductive_start_t = self.val_start_t + self.val_num_samples * self.args.t_unit
        self.test_inductive_start_t = self.test_transductive_start_t + self.test_transductive_num_samples * self.args.t_unit

        assert self.T == self.test_inductive_start_t + self.test_inductive_num_samples * self.args.t_unit, f"Total time should be equal to the sum of test inductive time and test inductive number of samples times time unit. Got {self.T} and {self.test_inductive_start_t + self.test_inductive_num_samples * self.args.t_unit} instead."
        assert self.total_num_samples == self.train_num_samples + self.val_num_samples + self.test_transductive_num_samples + self.test_inductive_num_samples, f"Total number of samples should be equal to the sum of train, val, test transductive, and test inductive number of samples. Got {self.total_num_samples} and {self.train_num_samples + self.val_num_samples + self.test_transductive_num_samples + self.test_inductive_num_samples} instead."

    @staticmethod
    def get_parser():
        parser = GraphGenerator.get_parser()
        parser.add_argument("--pattern-mode", type=str, choices=['er', 'nonlinear_er', 'nonlin_er_const_w_had', 'nonlin_er_const_w_dot', 'nonlin_er_a2_2nd_hop'], required=True)
        parser.add_argument("--val-ratio", type=float, required=True)
        parser.add_argument("--test-ratio", type=float, required=True)
        parser.add_argument("--test-inductive-ratio", type=float, required=True)
        parser.add_argument("--test-inductive-num-nodes-ratio", type=float, required=True)
        parser.add_argument("--visualize", action="store_true")

        # arguments specific for ER pattern
        parser.add_argument("--er-prob", type=float, required=True)
        parser.add_argument("--er-prob-inductive", type=float, required=True)

        # argument specific for nonlinear-ER pattern
            ## parser.add_argument("--er-prob", type=float, required=True)
            ## parser.add_argument("--er-prob-inductive", type=float, required=True)
        parser.add_argument("--mean", type=float, required=False, default=0.)
        parser.add_argument("--std", type=float, required=False, default=1.)

        # argument specific for nonlinear-ER-constant-weight-hadamard pattern
            ## parser.add_argument("--er-prob", type=float, required=True)
            ## parser.add_argument("--er-prob-inductive", type=float, required=True)
            ## parser.add_argument("--mean", type=float, required=False, default=0.)
            ## parser.add_argument("--std", type=float, required=False, default=1.)
        parser.add_argument("--weight-seed", default=5321, type=int)

        # argument specific for nonlinear-ER-constant-weight-dot pattern
            ## parser.add_argument("--er-prob", type=float, required=True)
            ## parser.add_argument("--er-prob-inductive", type=float, required=True)
            ## parser.add_argument("--mean", type=float, required=False, default=0.)
            ## parser.add_argument("--std", type=float, required=False, default=1.)
            ## parser.add_argument("--weight-seed", default=5321, type=int)

        # arguments specific for nonlinear-ER-2nd-hop pattern
            ## parser.add_argument("--er-prob", type=float, required=True)
            ## parser.add_argument("--er-prob-inductive", type=float, required=True)
        
        return parser

    def get_val_mask(self, t: np.ndarray) -> torch.Tensor:
        return np.logical_and(t >= self.val_start_t, t < self.test_transductive_start_t)

    def get_test_mask(self, t: np.ndarray) -> torch.Tensor:
        return np.logical_and(t >= self.test_transductive_start_t, t < self.test_inductive_start_t)
    
    def get_test_inductive_mask(self, t: np.ndarray) -> torch.Tensor:
        return t >= self.test_inductive_start_t
        
    def generate_pattern_graph(self, G_pattern: nx.Graph, transductive: bool = True) -> None:
        if self.args.pattern_mode in ["er", "nonlinear_er", "nonlin_er_const_w_had", "nonlin_er_const_w_dot", "nonlin_er_a2_2nd_hop"]:
            if transductive:
                p=self.args.er_prob
            else:
                p=self.args.er_prob_inductive
            er_graph = nx.erdos_renyi_graph(n=G_pattern.number_of_nodes(), p=p)
            G_pattern.add_edges_from(er_graph.edges(data=True))
            
            if self.args.pattern_mode == "nonlinear_er":
                for u, v in G_pattern.edges():
                    weight = np.random.normal(loc=self.args.mean, scale=self.args.std)
                    weight = max(weight, 0)
                    weight = round(weight, 3)
                    G_pattern[u][v]['weight'] = weight
        else:
            raise NotImplementedError()

    def generate_discovery_graph(self, discovery_G: nx.Graph, G_pattern_to_discover: nx.Graph) -> None:
        if G_pattern_to_discover.number_of_edges() == 0:
            return

        # Memory node (indexed 0) at t=t_n connects to nodes that have at least one edge in t=(t_n - K)
        if self.args.pattern_mode == "er":
            src_t, dst_t = nx_undirected_graph_to_sparse(G_pattern_to_discover)
            pattern_nodes = set(src_t).union(set(dst_t))
            discovery_G.add_edges_from([0, node_idx.item()] for node_idx in pattern_nodes)

        # Following variable stores per node if it has a positive connectivity. 
        # The goal is to bring non-linearity to the cause-and-effect dataset.
        elif self.args.pattern_mode == "nonlinear_er":
            src_t, dst_t = nx_undirected_graph_to_sparse(G_pattern_to_discover)
            pattern_nodes = set(src_t).union(set(dst_t))
            positive_connectivity = np.zeros((self.num_nodes), dtype=bool)
            for u, v, edge_data in G_pattern_to_discover.edges(data=True):
                # Skip and do not check if a node has already a positive connectivity.
                if positive_connectivity[u] > 0:
                    continue
                if edge_data['weight'] > self.args.mean:
                    positive_connectivity[u] = 1
                    positive_connectivity[v] = 1

            discovery_G.add_edges_from([0, node_idx.item()] for node_idx in pattern_nodes if positive_connectivity[node_idx])
        
        elif self.args.pattern_mode == "nonlin_er_const_w_had":
            src_t, dst_t = nx_undirected_graph_to_sparse(G_pattern_to_discover)
            rng = np.random.default_rng(seed=self.args.weight_seed)
            W = rng.normal(loc=self.args.mean, scale=self.args.std, size=(self.num_nodes, self.num_nodes))
            # Weight should be symmetric for undirected graphs.
            W = (W + W.T) / 2
            pattern_adj = np.zeros_like(W)
            pattern_adj[src_t, dst_t] = 1
            pattern_adj = ((pattern_adj * W) > 0).astype(np.uint8)
            pattern_src, pattern_dst = np.nonzero(pattern_adj)
            pattern_nodes = set(pattern_src).union(set(pattern_dst))
            
            discovery_G.add_edges_from([0, node_idx.item()] for node_idx in pattern_nodes)

        elif self.args.pattern_mode == "nonlin_er_const_w_dot":
            src_t, dst_t = nx_undirected_graph_to_sparse(G_pattern_to_discover)
            rng = np.random.default_rng(seed=self.args.weight_seed)
            W = rng.normal(loc=self.args.mean, scale=self.args.std, size=(self.num_nodes, self.num_nodes))
            # Weight should be symmetric for undirected graphs.
            W = (W + W.T) / 2
            pattern_adj = np.zeros_like(W)
            pattern_adj[src_t, dst_t] = 1
            pattern_adj = ((pattern_adj.dot(W)) > 0).astype(np.uint8)
            pattern_src, pattern_dst = np.nonzero(pattern_adj)
            pattern_nodes = set(pattern_src).union(set(pattern_dst))
            
            discovery_G.add_edges_from([0, node_idx.item()] for node_idx in pattern_nodes)

        elif self.args.pattern_mode == "nonlin_er_a2_2nd_hop":
            src_t, dst_t = nx_undirected_graph_to_sparse(G_pattern_to_discover)
            A = np.zeros((self.num_nodes, self.num_nodes))
            A[src_t, dst_t] = 1
            A = A.dot(A) - A
            A = (A > 0).astype(np.uint8)
            pattern_src, pattern_dst = np.nonzero(A)
            pattern_nodes = set(pattern_src).union(set(pattern_dst))

            discovery_G.add_edges_from([0, node_idx.item()] for node_idx in pattern_nodes)
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
            plt.figure(figsize=(20, 10))
            nx.draw_networkx(G, pos, node_size=40, with_labels=True, node_color="yellow")
            # Draw edge labels for the 'weight' attribute
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

            plt.title(f"Day {idx} out of {self.T}.")
            vis_save_dir = os.path.join(GraphGenerator.SAVE_DIR, self.dataset_name, "vis", stage)
            os.makedirs(vis_save_dir, exist_ok=True)
            plt.savefig(os.path.join(vis_save_dir, f"{idx}.png"))
            plt.close()
        

        with tqdm(total=self.total_num_samples, desc=self.dataset_name) as pbar:
            graph_patterns: List[nx.Graph] = []
            # Generate graph between t=0,...,k-1
            # No discovery; generating only a few pattern graphs at first.
            for i in range(self.gap):
                G: nx.Graph = nx.empty_graph(self.num_nodes)
                G_pattern: nx.Graph = nx.empty_graph(n=self.pattern_num_nodes_transductive)
                id1_pattern = id(G_pattern)
                self.generate_pattern_graph(G_pattern, transductive=True)
                id2_pattern = id(G_pattern)
                assert id1_pattern == id2_pattern

                node_mapping = {i: i + self.pattern_1st_node_idx_transductive for i in G_pattern.nodes}
                G_pattern = nx.relabel_nodes(G_pattern, node_mapping)
                G.add_edges_from(G_pattern.edges(data=True))
                graph_patterns.append(G_pattern)

                _update_sparse_data(G)
                if self.args.visualize:
                    _vis_graph(G, i, stage="t=0,...,k-1")

                now_t += self.args.t_unit
                pbar.update(1)

            # Generate graph between t=k,...,test_inductive_start_t
            # Train/val/test transductive sets are generated at this stage.
            for i in range(self.gap, self.total_num_samples - self.test_inductive_num_samples):
                G: nx.Graph = nx.empty_graph(self.num_nodes)

                G_new_pattern: nx.Graph = nx.empty_graph(n=self.pattern_num_nodes_transductive)
                id1_pattern = id(G_new_pattern)
                self.generate_pattern_graph(G_new_pattern, transductive=True)
                id2_pattern = id(G_new_pattern)
                assert id1_pattern == id2_pattern
                node_mapping = {i: i + self.pattern_1st_node_idx_transductive for i in G_new_pattern.nodes}
                G_new_pattern = nx.relabel_nodes(G_new_pattern, node_mapping)
                G.add_edges_from(G_new_pattern.edges(data=True))
                graph_patterns.append(G_new_pattern)

                G_pattern_to_discover = graph_patterns.pop(0)
                discovery_G: nx.Graph = nx.empty_graph(G.number_of_nodes())
                id1_discovery = id(discovery_G)
                self.generate_discovery_graph(discovery_G, G_pattern_to_discover)
                id2_discovery = id(discovery_G)
                assert id1_discovery == id2_discovery
                G.add_edges_from(discovery_G.edges(data=True))

                _update_sparse_data(G)
                if self.args.visualize:
                    if i < 4 * self.gap:
                        _vis_graph(G, i, stage="t=k,...,inductive_test_t")

                now_t += self.args.t_unit
                pbar.update(1)

            # Generate graph between t=test_inductive_start_t,...,T
            # Test inductive set is generated at this stage.
            for i in range(self.total_num_samples - self.test_inductive_num_samples, self.total_num_samples):
                G: nx.Graph = nx.empty_graph(self.num_nodes)

                G_new_pattern: nx.Graph = nx.empty_graph(n=self.pattern_num_nodes_inductive)
                id1_pattern = id(G_new_pattern)
                self.generate_pattern_graph(G_new_pattern, transductive=False)
                id2_pattern = id(G_new_pattern)
                assert id1_pattern == id2_pattern
                node_mapping = {i: i + self.pattern_1st_node_idx_inductive for i in G_new_pattern.nodes}
                G_new_pattern = nx.relabel_nodes(G_new_pattern, node_mapping)
                G.add_edges_from(G_new_pattern.edges(data=True))
                graph_patterns.append(G_new_pattern)

                G_pattern_to_discover = graph_patterns.pop(0)
                discovery_G: nx.Graph = nx.empty_graph(G.number_of_nodes())
                id1_discovery = id(discovery_G)
                self.generate_discovery_graph(discovery_G, G_pattern_to_discover)
                id2_discovery = id(discovery_G)
                assert id1_discovery == id2_discovery
                G.add_edges_from(discovery_G.edges(data=True))
                
                # To record empty snapshots, add a self-loop edge on node zero.
                if len(G.edges) == 0:
                    G.add_edge(0, 0)
                _update_sparse_data(G)
                if self.args.visualize:
                    if i < self.test_inductive_start_t + 3 * self.gap:
                        _vis_graph(G, i, stage="t=inductive_test_t,...,T")

                now_t += self.args.t_unit
                pbar.update(1)

        assert t.max() == self.T - self.args.t_unit, f"Last timestep should be {self.T - self.args.t_unit}. Got {t.max()} instead."

        return src.astype(self.node_datatype), dst.astype(self.node_datatype), t.astype(self.t_datatype), edge_feat.astype(self.edge_feat_datatype)