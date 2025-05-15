import torch

from ...DTDG.link_pred.trainer import LinkPredTrainer
from ...DTDG.gat import GATTrainer
from ...DTDG.trainer import NODE_EMB_MODEL_NAME

class LinkPredGATTrainer(LinkPredTrainer, GATTrainer):
    def before_training(self):
        pass

    def before_starting_window_training(self):
        pass
    
    def forward_backbone(self, snapshot: torch.Tensor, snapshot_feat: torch.Tensor, node_feat: torch.Tensor) -> torch.Tensor:
        src, dst = torch.nonzero(snapshot, as_tuple=True)    
        edge_index = torch.stack([src, dst], dim=0)
        edge_feat = snapshot_feat[src, dst]

        z = self.model[NODE_EMB_MODEL_NAME](
                node_feat,
                edge_index.long(),
                edge_weight=edge_feat)
        
        return z