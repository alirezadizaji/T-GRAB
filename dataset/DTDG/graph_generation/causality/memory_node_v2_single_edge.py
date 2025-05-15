"""
Version2 of memory node: odd days represent the edges between memory node and source nodes at previous time step; Even days represents a random generated edge between non-memory nodes.
"""

import random
import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.data import TemporalData

from ..negative_generator import NegativeEdgeGeneratorV2

def main():
    num_nodes = 15
    num_weeks = 14
    val_weeks = max(int(num_weeks * 0.15), 1)
    test_weeks = max(int(num_weeks * 0.15), 1)
    num_days = 7 * 2
    strategy = "rnd"
    num_neg_edge = None
    p_edge = 0.5
    name = f"mem-node-single-single-{num_weeks-val_weeks-test_weeks}w-n{num_nodes}"
    seed = 12345
    save_dir = "./TSA/data"

    T = num_weeks * num_days
    test_T = T - val_weeks * num_days
    val_T = test_T - test_weeks * num_days

    np.random.seed(seed)
    random.seed(seed)

    src = np.empty(0)
    dst = np.empty_like(src)
    t = np.empty_like(src)
    edge_feat = np.empty_like(src)

    # At snapshot 0, there shouldn't be any edge between memory node and the others
    G = nx.erdos_renyi_graph(n=num_nodes - 1, p=p_edge, directed=True)
    A = np.array(G.edges)
    A = A + 1
    idx = np.arange(A.shape[0])
    np.random.shuffle(idx)
    A = A[[idx[0]]]
    
    num_edges = A.shape[0]
    src = np.concatenate([src, A[:, 0]])
    dst = np.concatenate([dst, A[:, 1]])
    t = np.concatenate([t, np.repeat(0, num_edges)])
    edge_feat = np.concatenate([edge_feat, np.repeat(1, num_edges)])
    
    for i in range(1, T):
        new_src = A[:, 0]
        src = np.concatenate([src, A[:, 0]])
        dst = np.concatenate([dst, A[:, 1]])
        t = np.concatenate([t, np.full_like(new_src, fill_value=i)])
        edge_feat = np.concatenate([edge_feat, np.full_like(new_src, fill_value=1)])
    
    src = src.astype(np.int32)
    dst = dst.astype(np.int32)
    t = t.astype(np.int32)
    edge_feat = edge_feat.astype(np.int32)

    # Remove duplicate edges (if existed any)
    edges_t = np.stack([src, dst, t], axis=0).T     # N, 3
    edges_T = np.unique(edges_t, axis=0).T          # 3, N'
    src, dst, t = edges_T

    # Visualize first five snapshots
    pos = None
    for day in range(5):
        print(f"Day {day}")
        mask = (t == day)
        edges = np.stack([src[mask], dst[mask]], axis=0).T

        G = nx.DiGraph()
        G.add_nodes_from(np.arange(num_nodes))
        G.add_edges_from(edges)
        if pos is None:
            pos = nx.circular_layout(G)
        nx.draw_networkx(G, pos, node_size=40, with_labels=True, node_color="yellow")
        plt.title(f"Day {day}")
        plt.show(block=False)
        plt.pause(5)
        plt.close()

    test_mask = t >= test_T
    val_mask = np.logical_and(t < test_T, t >= val_T)
    train_mask = t < val_T

    print(f"\t Data generated. src shape: {src.shape}, edge_feat shape: {edge_feat.shape}", flush=True)

    data = TemporalData(
        src=torch.tensor(src),
        dst=torch.tensor(dst),
        t=torch.tensor(t),
        msg=torch.tensor(edge_feat))
    
    data_splits = {}
    data_splits['train'] = data[torch.tensor(train_mask)]
    data_splits['val'] = data[torch.tensor(val_mask)]
    data_splits['test'] = data[torch.tensor(test_mask)]

    # Ensure to only sample actual destination nodes as negatives.
    min_dst_idx, max_dst_idx = int(data.dst.min()), int(data.dst.max())

    # After successfully loading the dataset...
    if strategy == "hist_rnd":
        historical_data = data_splits["train"]
    else:
        historical_data = None
    neg_generator = NegativeEdgeGeneratorV2(
        dataset_name=name,
        first_dst_id=min_dst_idx,
        last_dst_id=max_dst_idx,
        num_neg_e=num_neg_edge,
        strategy=strategy,
        rnd_seed=seed,
        historical_data=historical_data,
    )

    
    fdir = os.path.join(save_dir, name)

    # generate validation negative edge set
    os.makedirs(fdir, exist_ok=True)
    import time
    start_time = time.time()
    split_mode = "val"
    print(
        f"INFO: Start generating negative samples: {split_mode} --- {strategy}"
    )
    neg_generator.generate_negative_samples(
        data=data_splits[split_mode], split_mode=split_mode, partial_path=fdir
    )
    print(
        f"INFO: End of negative samples generation. Elapsed Time (s): {time.time() - start_time: .4f}"
    )

    # generate test negative edge set
    start_time = time.time()
    split_mode = "test"
    print(
        f"INFO: Start generating negative samples: {split_mode} --- {strategy}"
    )
    neg_generator.generate_negative_samples(
        data=data_splits[split_mode], split_mode=split_mode, partial_path=fdir
    )
    print(
        f"INFO: End of negative samples generation. Elapsed Time (s): {time.time() - start_time: .4f}"
    )

    np.savez_compressed(os.path.join(fdir, "data.npz"), 
        src=src, 
        dst=dst, 
        t=t,
        edge_feat=edge_feat,
        num_nodes=num_nodes,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask)



if __name__ == "__main__":
    main()
