import gc
import os
from typing import List, Tuple, Union

import numpy as np
import torch
from torch.utils.data import Dataset as torchDataset

from .....utils.node_feat import NodeFeatGenerator
from .....utils import NodeFeatType

class ContinuousTimeLinkPredNodeFeatureStaticDataset(torchDataset):
    def __init__(self, 
                 data_loc: str, 
                 data_name: str, 
                 split: str = "train", 
                 node_feat: NodeFeatType = NodeFeatType.CONSTANT,
                 node_feat_dim: int = 1):
        """ Link prediction dataset with static node features. node features are not changed through the time and is same among all snapshots. """
        
        super(ContinuousTimeLinkPredNodeFeatureStaticDataset, self).__init__()
        self.split = split
        assert split in ["all", "train", "val", "test", "test_inductive"], f"`split` value should be among `all`, `train`, `val`, `test`, and `test_inductive`; got `{split}` instead."
        assert os.path.exists(data_loc), f"The given data location does not exist: {data_loc}"
        data_np = np.load(os.path.join(data_loc, data_name, "data.npz"))
        # NPZ file does not allow editing. the numpy data needs to be converted into the python dict first.
        data_np = {key: data_np[key] for key in data_np}

        kwds = ["src", "dst", "t", "edge_feat", "train_mask", "val_mask", "test_mask"]
        for k, complete_kwd_explanation in zip(kwds,
                                                ["source node id", "destination node id", "timestamp", "Edge features", "validation mask", "test mask"]):
            assert k in data_np, f"`{complete_kwd_explanation}`(`{k}`) is missing in numpy data."
        
        if "edge_ids" not in data_np:
            edge_id = 0
            edge_ids = np.empty_like(data_np["src"])
            map_edge_index_to_id = {}
            for i, (u, v, t) in enumerate(zip(data_np["src"], data_np["dst"], data_np["t"])):
                if (u, v, t) not in map_edge_index_to_id:
                    map_edge_index_to_id[(u, v, t)] = edge_id
                    map_edge_index_to_id[(v, u, t)] = edge_id
                    edge_id += 1
                edge_ids[i] = map_edge_index_to_id[(u, v, t)]
            data_np["edge_ids"] = edge_ids

        # Sort links based on time
        _, sorted_idx = torch.sort(torch.tensor(data_np["t"]), descending=False)
        sorted_idx = sorted_idx.cpu().numpy()
        for k in kwds + ["edge_ids"]:
            data_np[k] = data_np[k][sorted_idx]

        t = data_np["t"]
        src = data_np["src"]
        dst = data_np["dst"]
        edge_feat = data_np["edge_feat"]
        edge_ids = data_np["edge_ids"]

        self.num_nodes = int(data_np["num_nodes"])

        # Node features; shape: (num of nodes, node feat dim)
        self._node_feat = NodeFeatGenerator(node_feat)(self.num_nodes, node_feat_dim)
        # Edge features; shape: (num of edges, edge feat dim)
        edge_feat = data_np["edge_feat"]
        if edge_feat.ndim == 1:
            edge_feat = edge_feat[:, None]

        split_mask = f"{split}_mask"
        if split == "all":
            mask = np.ones_like(t, dtype=bool)
        elif split == "test_inductive":
            if split_mask not in data_np:
                print("!!! WARNING: 'test_inductive' does not exist in splits. An empty dataset is created.")
                mask = np.zeros_like(t, dtype=bool)
            else:
                mask = data_np[split_mask]
        else:
            mask = data_np[split_mask]

        del data_np
        gc.collect()

        self.t = torch.tensor(t[mask])
        self.src = torch.tensor(src[mask])
        self.dst = torch.tensor(dst[mask])
        self.edge_feat = torch.tensor(edge_feat[mask])
        self.edge_ids = torch.tensor(edge_ids[mask])

        # Test-inductive split might not exist.
        if self.t.numel() > 0:
            values, _ = torch.topk(self.t.unique(), k=2, largest=False)
            self.start_t = values.min().item()
            # By assuming the time is equally spaced.
            self.unit_t = torch.abs(values[0] - values[1]).item()

    def __len__(self):
        return self.t.numel()
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.src[index].long(), self.dst[index].long(), self.t[index], \
            self.edge_ids[index], self.edge_feat[index]

    def get_attr(self, 
                 query_src: torch.Tensor, 
                 query_dst: torch.Tensor, 
                 query_t: torch.Tensor,
                 attrs: Union[str, List[str]]="edge_ids") -> torch.Tensor:
        
        mask = ((self.src[:, None] == query_src[None, :].to(self.src.device)) & \
               (self.dst[:, None] == query_dst[None, :].to(self.src.device)) & \
               (self.t[:, None] == query_t[None, :].to(self.src.device)))
        
        idx_to_origin, idx_to_query = torch.nonzero(mask, as_tuple=True)
        idx_to_query = idx_to_query.to(query_src.device)

        if isinstance(attrs, str):
            attrs = [attrs]

        idx_not_filled_in_query = torch.ones_like(query_src)
        idx_not_filled_in_query[idx_to_query] = 0

        # If there is at least one invalid query, then raise the exception (not-existed)
        if torch.any(idx_not_filled_in_query):
            raise ValueError("Invalid query!")
        else:
            query_attrs = []
            for attr in attrs:
                origin_attr = getattr(self, attr)
                query_attr = torch.empty((query_src.shape[0], *origin_attr.shape[1:]), dtype=origin_attr.dtype, device=query_src.device)
                query_attr[idx_to_query] = origin_attr[idx_to_origin].to(query_src.device)
                query_attrs.append(query_attr)
        
        return *query_attrs,
