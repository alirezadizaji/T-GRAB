from abc import abstractmethod
import copy
import os
import re
from tqdm import tqdm
import pickle
from typing import Tuple

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch

from .graph_generator import GraphGenerator, nx_undirected_graph_to_sparse

import random


class PeriodicGenerator(GraphGenerator):
    _k_n_pattern = r"^\((\d+),\s*(\d+)\)$"
    _free_pattern = r"^(\d+(?: \d+)*)$"
    _multiple_pattern = r'\b(([0-9]\d*|\*)(?:x[1-9]\d*)?)(?:\s([0-9]\d*|\*)(?:x[1-9]\d*)?)*\b'

    PERIODIC_REGEX = f"{_k_n_pattern}|{_free_pattern}|{_multiple_pattern}"

    def __init__(self, 
                args) -> None:

        super(PeriodicGenerator, self).__init__(args.num_nodes,
                 args.dataset_name,
                 args.seed)

        assert re.search(PeriodicGenerator.PERIODIC_REGEX, self.dataset_name), f"Dataset name `{self.dataset_name}` doesn't follow the periodic regular expression."
        # Concat train/val/test number of weeks to the dataset name.
        self.dataset_name = self.dataset_name + f"/{args.topology_mode}-{self.num_nodes}n-{args.num_of_training_weeks}trW-{args.num_of_valid_weeks}vW-{args.num_of_test_weeks}tsW"        

        if args.topology_mode == 'fixed_er':
            self.dataset_name = self.dataset_name + f"-fp{args.fixed_er_prob}"
        elif args.topology_mode == 'sbm':
            self.dataset_name = self.dataset_name + f"-nc{args.num_clusters}-icp{args.intra_cluster_prob}-incp{args.inter_cluster_prob}"
        else:
            raise NotImplementedError()
        
        self.num_of_training_weeks = args.num_of_training_weeks
        self.num_of_valid_weeks = args.num_of_valid_weeks
        self.num_of_test_weeks = args.num_of_test_weeks
        self.num_of_periods = self.num_of_training_weeks + self.num_of_valid_weeks + self.num_of_test_weeks
        self.visualize = args.visualize

        # Define the regex pattern to capture the list of patterns appearing in a period
        self._patterns_in_a_period_list = []
        patterns_in_a_period = self.dataset_name.split("/")[0]
        if re.fullmatch(PeriodicGenerator._free_pattern, patterns_in_a_period):
            pattern = r"(\d+)"            
            self._patterns_in_a_period_list = re.findall(pattern, patterns_in_a_period)
        elif re.fullmatch(PeriodicGenerator._k_n_pattern, patterns_in_a_period):
            match = re.fullmatch(PeriodicGenerator._k_n_pattern, patterns_in_a_period)
            k = int(match.group(1))
            n = int(match.group(2))
            self._patterns_in_a_period_list = [str(ki) for ki in range(1, k + 1) for _ in range(n)]
        elif re.fullmatch(PeriodicGenerator._multiple_pattern, patterns_in_a_period):
            pattern = r'([0-9]\d*|\*)(?:x([1-9]\d*))?'
            patterns_in_a_period = self.dataset_name.split("/")[0]
            matches = re.findall(pattern, patterns_in_a_period)
            self._patterns_in_a_period_list = []
            for number, repetition in matches:
                count = int(repetition) if repetition else 1
                self._patterns_in_a_period_list.extend([number] * count)

        self.T = self.num_of_periods * len(self.patterns_in_a_period_list)

        # src and dst nodes data types.
        if self.num_nodes < 2**8:
            node_datatype = np.int16
        elif self.num_nodes < 2**16:
            node_datatype = np.int32
        else:
            node_datatype = np.int64
        self.node_datatype = node_datatype
        
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

        
    @property
    def patterns_in_a_period_list(self):
        return self._patterns_in_a_period_list

    @staticmethod
    def get_parser():
        parser = GraphGenerator.get_parser()
        parser.add_argument('-ntr', '--num-of-training-weeks', type=int, required=True)
        parser.add_argument('-nval', '--num-of-valid-weeks', type=int, required=True)
        parser.add_argument('-ntest', '--num-of-test-weeks', type=int, required=True)
        parser.add_argument('-v', '--visualize', action='store_true', help='If given, then the first two periods are visualized.')

        parser.add_argument('-top', '--topology-mode', type=str, required=True, choices=['fixed_er', 'sbm'])
        
        # fixed_er
        parser.add_argument('--fixed-er-prob', type=float, required=False)

        # sbm topology
        parser.add_argument('--num-clusters', type=int, required=False, help='Number of clusters for SBM mode')
        parser.add_argument('--intra-cluster-prob', type=float, required=False, help='List of intra-cluster edge probabilities for SBM mode')
        parser.add_argument('--inter-cluster-prob', type=float, required=False, help='Probability of edges between clusters for SBM mode')

        return parser

    def get_val_mask(self, t: np.ndarray) -> torch.Tensor:
        test_period = self.num_of_test_weeks * len(self.patterns_in_a_period_list)
        val_period = self.num_of_valid_weeks * len(self.patterns_in_a_period_list)
        mask = np.logical_and(t >= (self.T - test_period - val_period), t < (self.T - test_period))
        return mask

    def get_test_mask(self, t: np.ndarray) -> torch.Tensor:
        test_period = self.num_of_test_weeks * len(self.patterns_in_a_period_list)
        mask = (t >= self.T - test_period)
        return mask
    
    
    def generate_snapshot(self, G: nx.Graph, pattern: str, args_dict: dict, first_period: bool=False) -> nx.Graph:
        assert pattern.isdigit(), f"Pattern {pattern} for dataset generation {self.dataset_name} should be a digit!"
        pattern = int(pattern)
        num_nodes = args_dict['num_nodes']
        mode = args_dict['topology_mode']

        if mode == 'fixed_er':
            np.random.seed(12345+pattern)
            random.seed(12345+pattern)

            sample_G: nx.Graph = nx.erdos_renyi_graph(num_nodes, args_dict['fixed_er_prob'])
            G.add_edges_from(sample_G.edges)
        
        # SBM: periodic community assignment while the edge assignment is always stochastic.
        # Number of clusters is fixed.
        elif mode == 'sbm':
            num_clusters = int(args_dict['num_clusters'])
            
            base_count = self.num_nodes // num_clusters
            remained = self.num_nodes % num_clusters
            cluster_sizes = [base_count] * num_clusters
            for i in range(remained):
                cluster_sizes[i] += 1
            
            intra_cluster_prob = args_dict['intra_cluster_prob']  # List of probabilities for intra-cluster edges
            inter_cluster_prob = args_dict['inter_cluster_prob']  # Probability of edges between clusters

            # Defining sizes of clusters
            sizes = cluster_sizes
            total_nodes = sum(sizes)
            if total_nodes != self.num_nodes:
                raise ValueError("Total number of nodes in clusters exceeds the number of nodes available")

            # Defining the probability matrix, adjusted for different cluster sizes
            p_matrix = np.full((num_clusters, num_clusters), inter_cluster_prob)
            np.fill_diagonal(p_matrix, intra_cluster_prob)

            # Generate the SBM graph
            sbm_G: nx.Graph = nx.stochastic_block_model(sizes, p_matrix)
            
            # Periodically (and randomly) assign a node to a community 
            original_ids = list(sbm_G.nodes)
            random_ids = copy.deepcopy(original_ids)
            rng = np.random.Generator(np.random.PCG64(12345+pattern))
            rng.shuffle(random_ids)

            mapping = dict(zip(original_ids, random_ids))
            sample_G = nx.relabel_nodes(sbm_G, mapping)

        
        G.add_edges_from(sample_G.edges)

        return G

        

    def get_links(self, args_dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        now_t = 0

        src = np.empty(0, dtype=self.node_datatype)
        dst = np.empty_like(src)
        t = np.empty_like(src, dtype=self.t_datatype)
        edge_feat = np.empty_like(src, dtype=self.edge_feat_datatype)

        pos = None
        
        with tqdm(total=self.num_of_periods * len(self.patterns_in_a_period_list), desc=self.dataset_name.split("/")[1]) as pbar:
            for i in range(self.num_of_periods):
                for idx, pattern in enumerate(self.patterns_in_a_period_list):
                    G: nx.Graph = nx.empty_graph(self.num_nodes)
                    
                    id1 = id(G)
                    G = self.generate_snapshot(G, pattern, args_dict, i == 0)
                    id2 = id(G)
                    assert id1 == id2, "G variable should not be instantiated inside the generate_snapshot."
                    
                    if i <= 1 and self.visualize:
                        if pos is None:
                            pos = nx.circular_layout(G)
                        plt.figure(figsize=(20, 10))
                        nx.draw_networkx(G, pos, node_size=40, with_labels=True, node_color="yellow")
                        plt.title(f"Day {idx} out of {len(self.patterns_in_a_period_list)} with pattern {pattern}")
                        vis_save_dir = os.path.join(args_dict['save_dir'], self.dataset_name, "vis")
                        os.makedirs(vis_save_dir, exist_ok=True)
                        plt.savefig(os.path.join(vis_save_dir, f"{i}w_{idx}p.png"))
                        plt.close()
                    
                    # To prevent skipping timesteps which represent empty graphs, let's add only one self-loop edge on the node zero.
                    if len(G.edges) == 0:
                        G.add_edge(0, 0)

                    src_t, dst_t, edge_feat_t = nx_undirected_graph_to_sparse(G, return_edge_feat=True)
                    src = np.concatenate([src, src_t])
                    dst = np.concatenate([dst, dst_t])
                    t = np.concatenate([t, np.full_like(src_t, fill_value=now_t)])
                    edge_feat = np.concatenate([edge_feat, edge_feat_t])

                    now_t += 1
                    pbar.update(1)

        assert t.max() == self.T - 1, f"Last timestep should be {self.T - 1}. Got {t.max()} instead."
        
        return src, dst, t, edge_feat


    def create_data(self, args_dict) -> None:
        super(PeriodicGenerator, self).create_data(args_dict)
        
        fdir = os.path.join(args_dict['save_dir'], self.dataset_name)        
        with open(os.path.join(fdir, "patterns_in_a_period.pkl"), "wb") as f:
            pickle.dump(self.patterns_in_a_period_list, f)


