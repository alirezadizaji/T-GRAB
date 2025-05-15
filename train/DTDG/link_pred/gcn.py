import torch

from ...DTDG.link_pred.trainer import LinkPredTrainer
from ...DTDG.gcn import GCNTrainer
from ...DTDG.trainer import NODE_EMB_MODEL_NAME

class LinkPredGCNTrainer(LinkPredTrainer, GCNTrainer):
    def before_training(self):
        pass

    def before_starting_window_training(self):
        pass
    
    def forward_backbone(self, snapshot: torch.Tensor, snapshot_feat: torch.Tensor, node_feat: torch.Tensor) -> torch.Tensor:
        src, dst = torch.nonzero(snapshot, as_tuple=True)  
        edge_feat = snapshot_feat[src, dst]  
        edge_index = torch.stack([src, dst], dim=0)

        z = self.model[NODE_EMB_MODEL_NAME](
                node_feat,
                edge_index.long(),
                edge_weight=edge_feat)
        
        return z