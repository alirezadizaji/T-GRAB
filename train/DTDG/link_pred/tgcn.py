import torch

from ...DTDG.link_pred.trainer import LinkPredTrainer
from ...DTDG.tgcn import TGCNTrainer
from ...DTDG.trainer import NODE_EMB_MODEL_NAME

class LinkPredTGCNTrainer(LinkPredTrainer, TGCNTrainer):
    def before_training(self):
        "Initialize hidden state of GRU"
        self._h0 = torch.zeros((self.args.num_units, self.train_loader.dataset.num_nodes, self.args.out_channels)).to(self.device)
    
    def before_starting_window_training(self):
        # For every EvolveGCNO unit, reset the weights.
        self._h0 = self._h0.detach()

    def forward_backbone(self, snapshot: torch.Tensor, snapshot_feat: torch.Tensor, node_feat: torch.Tensor) -> torch.Tensor:
        # During evaluation mode, set h0 and c0 same way as before starting in train mode.
        if self.args.eval_mode and not hasattr(self, "_h0"):
            self.before_training()
        
        src, dst = torch.nonzero(snapshot, as_tuple=True)    
        edge_index = torch.stack([src, dst], dim=0)
        edge_feat = snapshot_feat[src, dst]

        h1, h = self.model[NODE_EMB_MODEL_NAME](
                node_feat,
                edge_index.long(),
                self._h0,
                edge_weight=edge_feat)
        
        self._h0 = h1
        z = h
        return z