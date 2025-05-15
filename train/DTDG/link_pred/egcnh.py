import torch

from ...DTDG.link_pred.trainer import LinkPredTrainer
from ...DTDG.egcnh import EvolveGCNTrainer
from ...DTDG.trainer import NODE_EMB_MODEL_NAME

class LinkPredEvolveGCNTrainer(LinkPredTrainer, EvolveGCNTrainer):
    def before_training(self):
        pass

    def before_starting_window_training(self):
        pass
    
    @property
    def retain_graph(self) -> bool:
        return True
    
    def forward_backbone(self, snapshot: torch.Tensor, snapshot_feat: torch.Tensor, node_feat: torch.Tensor) -> torch.Tensor:
        prev_src, prev_dst = torch.nonzero(snapshot, as_tuple=True)    
        prev_edge_index = torch.stack([prev_src, prev_dst], dim=0)
        prev_edge_weight = snapshot_feat[prev_src, prev_dst]

        z = self.model[NODE_EMB_MODEL_NAME](
                node_feat,
                prev_edge_index.long(),
                prev_edge_weight)
        
        return z