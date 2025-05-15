from typing import Optional, Tuple

import torch
import sys

from ...CTDG.link_pred.trainer import LinkPredTrainer
from ...CTDG.tgn_tgb import TGNTrainer
from ...CTDG.trainer import NODE_EMB_MODEL_NAME

class LinkPredTGNTrainer(LinkPredTrainer, TGNTrainer):
    def __init__(self):
        super().__init__()
    
    def before_epoch_training(self): 
        self.neighbor_loader.reset_state()
        self.model['memory'].train()
        self.model['memory'].reset_state()

    def after_iteration_training(self):
        self.model['memory'].detach()

    def after_epoch_training(self):
        pass

    def before_epoch_evaluation(self, split_mode: str):
        if split_mode == 'train':
            self.before_epoch_training()
        else:
            self.model['memory'].eval()

    def after_iteration_evaluation(self, split_mode):
        pass
    
    def after_epoch_evaluation(self, split_mode: str):
        pass

    def forward_backbone(self, 
                         batch_src: torch.Tensor, 
                         batch_dst: torch.Tensor, 
                         batch_t: torch.Tensor, 
                         batch_edge_id: torch.Tensor, 
                         batch_edge_feat: torch.Tensor,
                         batch_neg: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if batch_neg is not None:
            batch_neg_dst = batch_neg[1]
            n_id = torch.cat([batch_src, batch_dst, batch_neg_dst]).unique()
        else:
            n_id = torch.cat([batch_src, batch_dst]).unique()
        
        n_id, edge_index, e_id = self.neighbor_loader(n_id)
        self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)
        
        z, last_update = self.model['memory'](n_id)

        # Inject node features
        feats = self.model['memory'].node_raw_features[n_id]
        z = z + feats

        z = self.model[NODE_EMB_MODEL_NAME](z,
                                            last_update,
                                            edge_index,
                                            self.t[e_id],
                                            self.edge_feats[e_id])
        
        batch_src_node_embeddings, batch_dst_node_embeddings = z[self.assoc[batch_src]], z[self.assoc[batch_dst]]
        if batch_neg_dst is None:
            batch_neg_src_node_embeddings = None
            batch_neg_dst_node_embeddings = None
        else:
            batch_neg_src = batch_neg[0]
            batch_neg_dst = batch_neg[1]
            batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = z[self.assoc[batch_neg_src]], z[self.assoc[batch_neg_dst]]

        self.model['memory'].update_state(batch_src, batch_dst, batch_t, batch_edge_feat.float())
        self.neighbor_loader.insert(
            batch_src,
            batch_dst
        )

        return (batch_src_node_embeddings, batch_dst_node_embeddings), (batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings)