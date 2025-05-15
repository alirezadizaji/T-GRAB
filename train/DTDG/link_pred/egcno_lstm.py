import torch

from ...DTDG.link_pred.trainer import LinkPredTrainer
from ...DTDG.egcno_lstm import EvolveGCNLSTMTrainer
from ...DTDG.trainer import NODE_EMB_MODEL_NAME

class LinkPredEvolveGCNLSTMTrainer(LinkPredTrainer, EvolveGCNLSTMTrainer):
    def before_training(self):
        pass

    def before_starting_window_training(self):
        pass

    def forward_backbone(self, snapshot: torch.Tensor, node_feat: torch.Tensor) -> torch.Tensor:
        prev_src, prev_dst = torch.nonzero(snapshot, as_tuple=True)    
        prev_edge_index = torch.stack([prev_src, prev_dst], dim=0)
        z = self.model[NODE_EMB_MODEL_NAME](
                node_feat,
                prev_edge_index.long())
        
        return z
    
