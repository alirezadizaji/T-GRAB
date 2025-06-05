from abc import abstractmethod
import argparse
import gc
import random
from typing import Optional, Tuple

import os

import networkx as nx
import numpy as np
import torch
from torch_geometric.data import TemporalData

from ...utils.negative_generator import NegativeEdgeGeneratorV2


def nx_undirected_graph_to_sparse(G: nx.Graph, return_edge_feat: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    if G.number_of_edges() == 0:
        src_t = np.empty(0)
        dst_t = np.empty(0)
        if not return_edge_feat:
            return src_t, dst_t
        edge_feat_t = np.empty(0)
        return src_t, dst_t, edge_feat_t
    
    src_to_dst = np.array(G.edges)
    dst_to_src = src_to_dst[:, [1, 0]]
    edge_index = torch.tensor(np.concatenate([src_to_dst, dst_to_src]))
    src_t, dst_t = edge_index.T
    
    if not return_edge_feat:
        return src_t, dst_t
    
    edge_feat_t = torch.ones(edge_index.shape[0])
    for i, (u, v) in enumerate(edge_index):
        if 'weight' in G[u.item()][v.item()]:
            edge_feat_t[i] = G[u.item()][v.item()]['weight']
    
    return src_t, dst_t, edge_feat_t


class GraphGenerator:
    def __init__(self, 
                 num_nodes: int,
                 dataset_name: str,
                 seed: int = 12345):
        
        self.num_nodes = num_nodes
        self.dataset_name = dataset_name
        self.seed = seed

        # Set seed
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)

    @staticmethod
    def get_parser():
        parser = argparse.ArgumentParser('*** Dynamic Graph Generator ***', add_help=False)
        parser.add_argument('--num-nodes', required=True, type=int, help="Number of nodes each snapshot will have.")
        parser.add_argument('--dataset-name', required=True, type=str, help="The name of dataset to create.")
        parser.add_argument('--seed', required=True, type=int, default=12345)
        parser.add_argument('--save-dir', required=True, type=str, help="The directory to save the dataset.")

        return parser
    
    @abstractmethod
    def get_links(self, args_dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass

    @abstractmethod
    def get_val_mask(self, t: np.ndarray) -> torch.Tensor:
        pass

    @abstractmethod
    def get_test_mask(self, t: np.ndarray) -> torch.Tensor:
        pass

    def get_test_inductive_mask(self, t:np.ndarray) -> torch.Tensor:
        return np.zeros_like(t, dtype=bool)
    
    def create_data(self, args_dict) -> None:
        src, dst, t, edge_feat = self.get_links(args_dict)
        gc.collect()

        assert src.ndim == dst.ndim == t.ndim == edge_feat.ndim, f"Given src ({src.ndim}), dst ({dst.ndim}), t ({t.ndim}), and edge-feature ({edge_feat.ndim - 1}) should have the same number of dimensions."
        assert src.size == dst.size == t.size == edge_feat.size, f"There is mismatch number of elements among src ({src.size}), dst ({dst.size}), t ({t.size}), and edge features ({edge_feat.size})."

        val_mask = self.get_val_mask(t).astype(bool)
        test_mask = self.get_test_mask(t).astype(bool)
        test_inductive_mask = self.get_test_inductive_mask(t).astype(bool)

        assert val_mask.size == test_mask.size == test_inductive_mask.size == src.size, f"There is mismatch number of elements among src ({src.size}), validation mask ({val_mask.size}), and test mask ({test_mask.size})."
        train_mask = np.not_equal(np.ones_like(val_mask, dtype=bool), (val_mask | test_mask | test_inductive_mask))

        print(f"\t Data generated. src shape: {src.shape}, edge_feat shape: {edge_feat.shape}\n\tNumber of validation links: {val_mask.sum()}, test: {test_mask.sum()}, test_inductive: {test_inductive_mask.sum()}", flush=True)
        
        fdir = os.path.join(args_dict['save_dir'], self.dataset_name)

        # generate validation negative edge set
        os.makedirs(fdir, exist_ok=True)
            
        save_dir = os.path.join(fdir, "data.npz")
        np.savez(os.path.join(fdir, "data.npz"), 
            src=src, 
            dst=dst, 
            t=t,
            edge_feat=edge_feat,
            num_nodes=self.num_nodes,
            train_mask=train_mask,
            val_mask=val_mask,
            test_inductive_mask=test_inductive_mask,
            test_mask=test_mask)
        print(f"@@@ Data saved at {save_dir}", flush=True)
