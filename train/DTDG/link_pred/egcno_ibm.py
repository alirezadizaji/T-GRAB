import numpy as np
import torch

from ...DTDG.link_pred.trainer import LinkPredTrainer
from ...DTDG.egcno_ibm import EvolveGCNTrainer
from ...DTDG.trainer import NODE_EMB_MODEL_NAME

class LinkPredEvolveGCNTrainer(LinkPredTrainer, EvolveGCNTrainer):
    def before_training(self):
        pass

    def before_starting_window_training(self):
        pass
    
    def forward_backbone(self, snapshot: torch.Tensor, snapshot_feat: torch.Tensor, node_feat: torch.Tensor) -> torch.Tensor:
        # src, dst = torch.nonzero(snapshot, as_tuple=True)    
        # edge_index = torch.stack([src, dst], dim=0)
        # edge_feat = snapshot_feat[src, dst]
        nodes_list = [node_feat]
        A_list = [snapshot.float()]
        z = self.model[NODE_EMB_MODEL_NAME](
                A_list,
                nodes_list)
        
        return z
    
