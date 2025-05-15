import gc
import os
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset as torchDataset

from ....CTDG.torch_dataset.link_pred.node_feat_static import ContinuousTimeLinkPredNodeFeatureStaticDataset
from .....utils import NodeFeatGenerator, NodeFeatType

class LinkPredNodeFeatureStaticDataset(ContinuousTimeLinkPredNodeFeatureStaticDataset):
    def __init__(self, 
                 data_loc: str, 
                 data_name: str, 
                 split: str = "train", 
                 node_feat: NodeFeatType = NodeFeatType.CONSTANT,
                 node_feat_dim: int = 1,
                 to_dense: bool = True,
                 return_edge_info: bool = False):
        """ Link prediction dataset with static node features. node features are not changed through the time and is same among all snapshots. """
        
        super(LinkPredNodeFeatureStaticDataset, self).__init__(data_loc, 
                                                               data_name, 
                                                               split, 
                                                               node_feat, 
                                                               node_feat_dim)
        
        # Make sure links are all sorted chronologically.
        assert torch.all(self.t[:-1] <= self.t[1:]), "Links should be sorted chronologically."

        # links are already sorted based on time.
        self.unique_t, unique_t_counts = torch.unique(self.t, return_counts=True)

        # this records the first occurence of each timestamp in `t`.
        self.t_first_occurence = torch.cat([torch.zeros(1), unique_t_counts.cumsum(0)]).long()

        # If True, then return edge_feat and edge_id as well.
        self.return_edge_info = return_edge_info

        # If true then return dense tensor in a batch, otherwise return sparse tensors.
        self.to_dense = to_dense

    def __len__(self):
        return len(self.unique_t)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ It returns a tuple of three: previous timestep snapshot, current timestep snapshot, node features of current snapshot. """
        if not isinstance(index, int):
            assert len(index) == 1, f"Currently this dataset is supported with batch size=1; got {len(index)} instead."
            index = index[0]
        
        # Index here represents timestamp
        # For all splits (train/val/test), index starts from 0.
        curr_t = self.start_t + index * self.unit_t

        if index == 0:
            prev_index = 0
        else:
            prev_index = index - 1

        prev_idx_start, prev_idx_end = self.t_first_occurence[prev_index].item(), self.t_first_occurence[prev_index + 1].item()
        prev_src = self.src[prev_idx_start:prev_idx_end]
        prev_dst = self.dst[prev_idx_start:prev_idx_end]
        prev_edge_feat = self.edge_feat[prev_idx_start:prev_idx_end]

        curr_idx_start, curr_idx_end = self.t_first_occurence[index].item(), self.t_first_occurence[index + 1].item()
        curr_src = self.src[curr_idx_start:curr_idx_end]
        curr_dst = self.dst[curr_idx_start:curr_idx_end]
        curr_edge_feat = self.edge_feat[curr_idx_start:curr_idx_end]
        curr_edge_id = self.edge_ids[curr_idx_start:curr_idx_end]

        if not self.return_edge_info:
            if self.to_dense:
                prev_snapshot = torch.zeros(self.num_nodes, self.num_nodes, dtype=bool)  
                prev_edge_feat_snapshot = torch.zeros((self.num_nodes, self.num_nodes, *self.edge_feat.shape[1:]), dtype=torch.float32)
                curr_snapshot = torch.zeros_like(prev_snapshot)
                curr_edge_feat_snapshot = torch.zeros_like(prev_edge_feat_snapshot)
        
                prev_snapshot[prev_src, prev_dst] = 1
                curr_snapshot[curr_src, curr_dst] = 1
                prev_edge_feat_snapshot[prev_src, prev_dst] = prev_edge_feat.float()
                curr_edge_feat_snapshot[curr_src, curr_dst] = curr_edge_feat.float()
            
                return prev_snapshot, prev_edge_feat_snapshot, curr_snapshot, curr_edge_feat_snapshot, self._node_feat, curr_t
            
            else:
                raise NotImplementedError("Not supported yet.")
                # return (prev_src, prev_dst), (curr_src, curr_dst), self._node_feat, curr_t
        else:
            if self.to_dense:
                raise NotImplementedError("Not supported yet.")
                # prev_snapshot = torch.zeros(self.num_nodes, self.num_nodes, dtype=bool)  
                # curr_snapshot = torch.zeros_like(prev_snapshot)
                # curr_snapshot_edge_feat = torch.zeros((*curr_snapshot.shape, *self.edge_feat.shape[1:]), dtype=torch.float)
                # curr_snapshot_edge_id = torch.zeros_like(curr_snapshot, dtype=torch.long)
            
                # prev_snapshot[prev_src, prev_dst] = 1
                # curr_snapshot[curr_src, curr_dst] = 1
                # curr_snapshot_edge_feat[curr_src, curr_dst] = curr_edge_feat
                # curr_snapshot_edge_id[curr_src, curr_dst] = curr_edge_id

                # return prev_snapshot, curr_snapshot, self._node_feat, curr_t, curr_snapshot_edge_feat, curr_snapshot_edge_id 
            
            else:
                return (curr_src, curr_dst), self._node_feat, curr_t, curr_edge_feat, curr_edge_id