import torch

from ...DTDG.link_pred.trainer import LinkPredTrainer
from ...DTDG.egcnh_batch_sep import EvolveGCNHTrainer
from ...DTDG.trainer import NODE_EMB_MODEL_NAME

class LinkPredEvolveGCNHTrainer(LinkPredTrainer, EvolveGCNHTrainer):
    def before_training(self):
        pass

    def before_starting_window_training(self):
        pass
 
    def forward_backbone(self, snapshot: torch.Tensor, node_feat: torch.Tensor) -> torch.Tensor:
        prev_src, prev_dst = torch.nonzero(snapshot, as_tuple=True)    
        prev_edge_index = torch.stack([prev_src, prev_dst], dim=0)
        if not hasattr(self, 'H'):
            self.H = torch.zeros((node_feat.shape[0], self.args.in_channels)).to(node_feat.device)
        H_cloned = self.H.detach().clone()
        self.H = self.model[NODE_EMB_MODEL_NAME](
                node_feat,
                H_cloned,
                prev_edge_index.long())
        
        return self.H
    
