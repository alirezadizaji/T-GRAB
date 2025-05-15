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

    PATTERN_MASK_FOLDER_NAME="pattern_masks"
    PATTERN_PICKLE_FILE="{}_mask.pkl"
    PATTERN_MASKS_LIST=[
        "subpattern",
        "whole_graph"
    ]

    def __init__(self, 
                #  num_of_training_weeks: int, 
                #  num_of_valid_weeks: int, 
                #  num_of_test_weeks: int,
                #  visualize: bool,
                #  top: str,
                #  *args, 
                #  **kwargs
                args
                 ) -> None:
        """ 
        Args:
            dataset_name (str): Dataset name should follow a specific `PeriodicGenerator.PERIODIC_REGEX`. This is important to make data generation compatible with evaluation during training. 
                
                Example: _k_n_pattern: (3, 4)/data1 which is equivalent to '1 1 1 1 2 2 2 2 3 3 3 3/data1'.
                         _free_pattern: 1 2 3 1 4 1 2 3 1 4/data2,
                         _multiple_pattern. 1 2x1000 3/data3,
                
                NOTE: periodic regex should follow this way:
                    1. before `/`, the pattern for one period appears; the pattern consists of numbers, and '*' (random graph).
                    2. To make the pattern modelling easier, it's possible to skip passing same consecutive number. Please checkout the Examples.
            
            num_of_training_weeks (str): Number of training weeks to generate periodic patterns
            num_of_valid_weeks (str): Number of validation weeks to generate periodic patterns
            num_of_test_weeks (str): Number of test weeks to generate periodic patterns
            directed (bool, Default: False): If False, then generate undirected snapshots, otherwise directed one.
        """
        super(PeriodicGenerator, self).__init__(args.num_nodes,
                 args.dataset_name,
                 args.neg_sampling_strategy, 
                 args.seed, 
                 args.num_neg_links_to_sample_per_pos_link,
                 args.do_neg_sampling)

        assert re.search(PeriodicGenerator.PERIODIC_REGEX, self.dataset_name), f"Dataset name `{self.dataset_name}` doesn't follow the periodic regular expression."
        # Concat train/val/test number of weeks to the dataset name.
        self.dataset_name = self.dataset_name + f"/{args.topology_mode}-{self.num_nodes}n-{args.num_of_training_weeks}trW-{args.num_of_valid_weeks}vW-{args.num_of_test_weeks}tsW"        

        if args.topology_mode == 'rotary_clique':
            self.dataset_name = self.dataset_name + f"-c{args.clique_size}-p{args.probability}-sk{args.skip_size}-fs{args.first_pattern_starting_index}"

        elif args.topology_mode == 'fixed_er':
            self.dataset_name = self.dataset_name + f"-p{args.probability}-fp{args.fixed_er_prob}"

        elif args.topology_mode == 'sbm':
            self.dataset_name = self.dataset_name + f"-p{args.probability}-nc{args.num_clusters}-cc{args.cluster_sizes}-icp{args.intra_cluster_prob}-incp{args.inter_cluster_prob}-am{args.additional_mode}-aicp{args.additional_intra_cluster_prob}-aep{args.additional_er_prob}"
        
        elif args.topology_mode == 'sbm_sto':
            self.dataset_name = self.dataset_name + f"-p{args.probability}-nc{args.num_clusters}-icp{args.intra_cluster_prob}-incp{args.inter_cluster_prob}"

        elif args.topology_mode == 'sbm_sto_v2':
            self.dataset_name = self.dataset_name + f"-p{args.probability}-nc{args.num_clusters}-icp{args.intra_cluster_prob}-incp{args.inter_cluster_prob}"

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

        self.pattern_mask = {k: {} for k in PeriodicGenerator.PATTERN_MASKS_LIST}

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

        parser.add_argument('-top', '--topology-mode', type=str, required=True, choices=['rotary_clique', 'fixed_er', 'sbm', 'sbm_sto', 'sbm_sto_v2'])
        # parser.add_argument('-top', '--fixed-mode', type=str, required=False, choices=['er', 'sbm'])
        
        parser.add_argument('-prob', '--probability', type=float, required=True, default=0)
        parser.add_argument('-prune-mode', '--pruning-mode', type=str, required=True, choices=['none', 'rotary_clique'])

        
        #clique topology
        parser.add_argument('--clique-size', type=int, required=False, default=None)
        parser.add_argument('--skip-size', type=int, required=False, default=None)
        parser.add_argument('--first-pattern-starting-index', type=int, required=False, default=None)

        # sbm topology
        parser.add_argument('--num-clusters', type=int, nargs='+', required=False, help='Number of clusters for SBM mode')
        parser.add_argument('--cluster-sizes', type=int, nargs='+', required=False, help='Sizes of each cluster for SBM mode (can be a list or an integer)')
        parser.add_argument('--intra-cluster-prob', type=float, required=False, help='List of intra-cluster edge probabilities for SBM mode')
        parser.add_argument('--inter-cluster-prob', type=float, required=False, help='Probability of edges between clusters for SBM mode')
        parser.add_argument('--additional-mode', type=str, required=False, choices=['empty', 'sbm', 'er'], help='Additional mode for remaining nodes in SBM')
        parser.add_argument('--additional-intra-cluster-prob', type=float, required=False, help='Intra-cluster probability for additional SBM cluster if additional mode is sbm')
        parser.add_argument('--additional-er-prob', type=float, required=False, help='Edge probability for additional Erdős–Rényi model if additional mode is er')

        # sbm stochastic topology
        ## --num-clusters
        ## --intra-cluster-prob
        ## inter-cluster-prob

        #circle topology

        #TODO add subparser

        # fixed_er
        parser.add_argument('--fixed-er-prob', type=float, required=False)

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
    
    
    def pruning_snapshot(self, param: dict, mode: str='none'):
        clique_size = param['clique_size']
        num_nodes = param['num_nodes']
        skip_size = param['skip_size']
        num_patterns = len(set(self.patterns_in_a_period_list))
        if 0 in self.patterns_in_a_period_list:
            num_patterns -= 1
        first_pattern_starting_index = param['first_pattern_starting_index'] 

        if mode == 'none':
            return []

        elif mode == 'rotary_clique':
            return [(u % num_nodes, v % num_nodes) for i in range(num_patterns) for u in range(first_pattern_starting_index + skip_size * i, first_pattern_starting_index + (i * skip_size) + clique_size)  \
                                                                                    for v in range(u, first_pattern_starting_index + (i * skip_size) + clique_size)]
        else:
            raise NotImplementedError()

    def generate_subpattern_mask(self, edge_index: np.ndarray) -> np.ndarray:
        assert edge_index.shape[1] == 2

        mask = np.zeros((self.num_nodes, self.num_nodes), dtype=np.uint8)
        mask[edge_index[:, 0], edge_index[:, 1]] = 1
        mask[edge_index[:, 1], edge_index[:, 0]] = 1
        np.fill_diagonal(mask, 0)

        return mask    
    
    def generate_snapshot(self, G: nx.Graph, pattern: str, args_dict: dict, first_period: bool=False) -> nx.Graph:
        assert pattern.isdigit(), f"Pattern {pattern} for dataset generation {self.dataset_name} should be a digit!"

        if 'fixed_er' == args_dict['topology_mode']:
            assert args_dict['probability'] == 0

        # Adding the random graph
        rand_G: nx.Graph = nx.erdos_renyi_graph(self.num_nodes, args_dict['probability'])
        G.add_edges_from(rand_G.edges)

        G.remove_edges_from(self.prune_edges)

        edge_index = self.generate_snapshot_topology(int(pattern), args_dict, mode=args_dict['topology_mode'])
        G.add_edges_from(edge_index)

        if first_period:
            subpattern_name = args_dict['topology_mode'] + "_" + pattern
            if subpattern_name not in self.pattern_mask['subpattern']: 
                # pattern 0 always represents no motif; basically an empty or random graph.
                if int(pattern) > 0:
                    self.pattern_mask['subpattern'][subpattern_name] = self.generate_subpattern_mask(np.array(edge_index))           
        
        return G

    def generate_snapshot_topology(self, pattern: int, param: dict, mode: str='rotary_clique'):
        if pattern == 0:
            return []
        num_nodes = param['num_nodes']

        if mode == 'rotary_clique':
            clique_size = param['clique_size']
            skip_size = param['skip_size']
            clique_starting_idx_node = param['first_pattern_starting_index']  + skip_size * (pattern - 1)
            return [(u % num_nodes, v % num_nodes) for u in range(clique_starting_idx_node, clique_starting_idx_node + clique_size) for v in range(u, clique_starting_idx_node + clique_size) if u != v]
        elif mode == 'fixed_er':
            np.random.seed(12345+pattern)
            random.seed(12345+pattern)

            rand_G: nx.Graph = nx.erdos_renyi_graph(num_nodes, param['fixed_er_prob'])

            return rand_G.edges
        
        # SBM deterministic: Periodic edge assignment with fixed community assignment
        elif mode == 'sbm':
            num_clusters = param.get('num_clusters', 2)
            cluster_sizes = param['cluster_sizes']
            intra_cluster_prob = param['intra_cluster_prob']  # List of probabilities for intra-cluster edges
            inter_cluster_prob = param['inter_cluster_prob']  # Probability of edges between clusters

            # Check if cluster_sizes is an integer or a list
            if isinstance(cluster_sizes, int):
                cluster_sizes = [cluster_sizes] * num_clusters
            elif len(cluster_sizes) != num_clusters:
                raise ValueError("Length of cluster_sizes must be equal to num_clusters")

            # Check if the length of intra_cluster_prob matches num_clusters
            if len(intra_cluster_prob) != num_clusters:
                raise ValueError("Length of intra_cluster_prob must be equal to num_clusters")

            # Defining sizes of clusters
            sizes = cluster_sizes
            total_nodes = sum(sizes)
            if total_nodes > num_nodes:
                raise ValueError("Total number of nodes in clusters exceeds the number of nodes available")

            # Defining the probability matrix, adjusted for different cluster sizes
            p_matrix = []
            for i in range(num_clusters):
                row = []
                for j in range(num_clusters):
                    if i == j:
                        row.append(intra_cluster_prob[i])
                    else:
                        row.append(inter_cluster_prob)
                        # Adjust inter-cluster probability based on cluster sizes
                        # row.append(inter_cluster_prob * (sizes[i] / total_nodes) * (sizes[j] / total_nodes))
                p_matrix.append(row)

            # Generate the SBM graph
            sbm_G: nx.Graph = nx.stochastic_block_model(sizes, p_matrix, seed=12345+pattern)

            # Option to add an additional component to the rest of the graph
            remaining_nodes = num_nodes - total_nodes
            if remaining_nodes > 0:
                additional_mode = param.get('additional_mode', 'empty')
                if additional_mode == 'empty':
                    pass  # Leave the rest of the graph empty
                elif additional_mode == 'sbm':
                    additional_cluster_size = remaining_nodes
                    additional_intra_prob = param['additional_intra_cluster_prob']
                    additional_sbm_G: nx.Graph = nx.stochastic_block_model([additional_cluster_size], [[additional_intra_prob]], seed=12345+pattern)
                    sbm_G.add_edges_from(additional_sbm_G.edges)
                elif additional_mode == 'er':
                    er_prob = param['additional_er_prob']
                    additional_er_G: nx.Graph = nx.erdos_renyi_graph(remaining_nodes, er_prob, seed=12345+pattern)
                    sbm_G.add_edges_from(additional_er_G.edges)
                else:
                    raise NotImplementedError(f"Additional mode '{additional_mode}' is not implemented.")

            return sbm_G.edges

        # SBM Stochastic: periodic community assignment while the edge assignment is always stochastic.
        # Number of clusters is periodic.
        elif mode == 'sbm_sto':
            num_clusters = int(param['num_clusters'][pattern - 1])
            
            base_count = self.num_nodes // num_clusters
            remained = self.num_nodes % num_clusters
            cluster_sizes = [base_count] * num_clusters
            for i in range(remained):
                cluster_sizes[i] += 1
            
            intra_cluster_prob = param['intra_cluster_prob']  # List of probabilities for intra-cluster edges
            inter_cluster_prob = param['inter_cluster_prob']  # Probability of edges between clusters

            # # Check if the length of intra_cluster_prob matches num_clusters
            # if len(intra_cluster_prob) != num_clusters:
            #     raise ValueError("Length of intra_cluster_prob must be equal to num_clusters")

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

            return sbm_G.edges
        
        # SBM Stochastic v2: periodic community assignment while the edge assignment is always stochastic.
        # Number of clusters is fixed.
        # Node assignment is periodic.
        elif mode == 'sbm_sto_v2':
            assert len(param['num_clusters']) == 1, "This version only supports one number of clusters."
            num_clusters = int(param['num_clusters'][0])
            
            base_count = self.num_nodes // num_clusters
            remained = self.num_nodes % num_clusters
            cluster_sizes = [base_count] * num_clusters
            for i in range(remained):
                cluster_sizes[i] += 1
            
            intra_cluster_prob = param['intra_cluster_prob']  # List of probabilities for intra-cluster edges
            inter_cluster_prob = param['inter_cluster_prob']  # Probability of edges between clusters

            # # Check if the length of intra_cluster_prob matches num_clusters
            # if len(intra_cluster_prob) != num_clusters:
            #     raise ValueError("Length of intra_cluster_prob must be equal to num_clusters")

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
            sbm_G_shuffled = nx.relabel_nodes(sbm_G, mapping)

            return sbm_G_shuffled.edges

        elif mode == 'circle':
            return []
        
        elif mode == 'er':
            return []
        
        else:
            raise NotImplementedError()

    def get_links(self, args_dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        now_t = 0

        src = np.empty(0, dtype=self.node_datatype)
        dst = np.empty_like(src)
        t = np.empty_like(src, dtype=self.t_datatype)
        edge_feat = np.empty_like(src, dtype=self.edge_feat_datatype)

        pos = None
        
        self.prune_edges = self.pruning_snapshot(args_dict, mode=args_dict['pruning_mode'])
        
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
                        vis_save_dir = os.path.join(GraphGenerator.SAVE_DIR, self.dataset_name, "vis")
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
        
        if self.pattern_mask is None:
            raise ValueError("subpattern_mask should not be None. Please checkout your function definition for `get_subpattern_mask`")
        
        fdir = os.path.join(GraphGenerator.SAVE_DIR, self.dataset_name)
        pattern_masks_dir = os.path.join(fdir, PeriodicGenerator.PATTERN_MASK_FOLDER_NAME)
        os.makedirs(pattern_masks_dir, exist_ok=True)
        for category in self.pattern_mask.keys():
            with open(os.path.join(pattern_masks_dir, PeriodicGenerator.PATTERN_PICKLE_FILE.format(category)), "wb") as f:
                pickle.dump(self.pattern_mask[category], f)
        
        with open(os.path.join(fdir, "patterns_in_a_period.pkl"), "wb") as f:
            pickle.dump(self.patterns_in_a_period_list, f)


