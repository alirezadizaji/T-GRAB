import os
import shutil
from typing_extensions import Protocol
from typing import Callable

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch


class SampleVisualizer(Protocol):
    def __call__(self, pred_adj: torch.Tensor, target_adj: torch.Tensor, split_mode: str, epoch: int, filename: str) -> None:
        """ It visualizes model prediction alongside with the ground-truth.
        """
        pass    

def visualizer(save_dir: str, 
               num_nodes: int, 
               node_pos="kamada_kawai_layout") -> SampleVisualizer:
    
    # This stores position of nodes once and then it's used for different snapshots.
    os.makedirs(save_dir, exist_ok=True)
    pos_ = [None]
    # best_save_dir = [None]

    def _visualize(pred_adj: torch.Tensor, target_adj: torch.Tensor, split_mode: str, epoch: int, filename: str):
        
        # # Deletes the visualization from previous epoch
        # if best_save_dir[0] is not None:
        #     prev_epoch = os.path.split(best_save_dir[0])[-1]
        #     if prev_epoch != epoch:
                # shutil.rmtree(best_save_dir[0])

        split_epoch_save_dir = os.path.join(save_dir, split_mode, str(epoch))
        # best_save_dir[0] = split_epoch_save_dir
        os.makedirs(split_epoch_save_dir, exist_ok=True)

        pred_adj = (pred_adj >= 0.5)
        
        if isinstance(pred_adj, torch.Tensor):
            pred_adj = pred_adj.detach().cpu().numpy()

        pred_src, pred_dst = np.nonzero(pred_adj)

        if isinstance(target_adj, torch.Tensor):
            target_adj = target_adj.detach().cpu().numpy()

        target_src, target_dst = np.nonzero(target_adj)

        G1 = nx.DiGraph()
        G1.add_nodes_from(list(range(num_nodes)))
        G1.add_edges_from(list(zip(target_src, target_dst)))
        if pos_[0] is None:
            func = getattr(nx, node_pos)
            pos_[0] = func(G1)
        _, axes = plt.subplots(1, 2, figsize=(20, 10))
        nx.draw_networkx(G1, pos_[0], node_size=200, node_color='lightblue', ax=axes[0])
        axes[0].set_title(f"Target", fontsize=30)
        G2 = nx.DiGraph()
        G2.add_nodes_from(list(range(num_nodes)))
        G2.add_edges_from(list(zip(pred_src, pred_dst)))
        nx.draw_networkx(G2, pos_[0], node_size=200, node_color='lightblue', ax=axes[1])
        axes[1].set_title(f"Prediction", fontsize=30)

        d = os.path.join(split_epoch_save_dir, "vis")
        os.makedirs(d, exist_ok=True)
        d = os.path.join(d, filename)
        plt.suptitle(filename, fontsize=40)
        plt.savefig(d)
        plt.close()

    return _visualize