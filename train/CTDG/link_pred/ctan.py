from typing import Optional, Tuple

import torch

from ...CTDG.link_pred.trainer import LinkPredTrainer
from ...CTDG.ctan import CTANTrainer
from ...CTDG.trainer import NODE_EMB_MODEL_NAME

class LinkPredCTANTrainer(LinkPredTrainer, CTANTrainer):
    def __init__(self):
        super().__init__()  
    
    def _update_memory_and_neighbor_loader(self):
        #TODO: setting batch_t as long() can cause some issues for links with non-integer timestamps.
        # Before the time to use long(), CTAN basically expects to pass a long batch_t to update the memory.
        self.model[NODE_EMB_MODEL_NAME].update(self.ctan_batch_src.long(), 
                                               self.ctan_batch_dst.long(), 
                                               self.ctan_batch_t.long(), 
                                               self.ctan_batch_edge_feat, 
                                               self.ctan_batch_src_node_embeddings,
                                               self.ctan_batch_dst_node_embeddings)
        
        self.neighbor_loader.insert(self.ctan_batch_src.long(), self.ctan_batch_dst.long())
               
    def before_epoch_training(self):  
        self.model[NODE_EMB_MODEL_NAME].reset_memory()
        self.neighbor_loader.reset_state()      

    def before_iteration_training(self):
        pass

    def after_iteration_training(self):
        self._update_memory_and_neighbor_loader()
        self.model[NODE_EMB_MODEL_NAME].detach_memory()

    def after_epoch_training(self):
        pass

    def before_epoch_evaluation(self, split_mode):
        if split_mode == "train":
            self.before_epoch_training()
        pass

    def after_iteration_evaluation(self, split_mode: str):
        self._update_memory_and_neighbor_loader()
        
    def after_epoch_evaluation(self, split_mode):
        pass

    def forward_backbone(self, 
                         batch_src: torch.Tensor, 
                         batch_dst: torch.Tensor, 
                         batch_t: torch.Tensor, 
                         batch_edge_id: torch.Tensor, 
                         batch_edge_feat: Optional[torch.Tensor],
                         batch_neg: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if batch_neg is not None:
            batch_neg_dst = batch_neg[1]
            n_id = torch.cat([batch_src, batch_dst, batch_neg_dst]).unique()
        else:
            n_id = torch.cat([batch_src, batch_dst]).unique()
        
        edge_index = torch.empty(size=(2,0)).long()
        e_id = self.neighbor_loader.e_id[n_id]
        for _ in range(self.model[NODE_EMB_MODEL_NAME].num_gnn_layers):
            n_id, edge_index, e_id = self.neighbor_loader(n_id)
        self.assoc[n_id] = torch.arange(n_id.size(0), device=self.device)
        
        z = self.model[NODE_EMB_MODEL_NAME](x=self.train_loader.dataset._node_feat.to(self.device),
                                            n_id=n_id,
                                            msg=self.full_dataset.edge_feat.to(self.device)[e_id],
                                            t=self.full_dataset.t.to(self.device)[e_id],
                                            edge_index=edge_index)

        batch_src_node_embeddings, batch_dst_node_embeddings = z[self.assoc[batch_src]], z[self.assoc[batch_dst]]

        self.ctan_batch_src = batch_src
        self.ctan_batch_dst = batch_dst
        self.ctan_batch_t = batch_t
        self.ctan_batch_edge_feat = batch_edge_feat
        self.ctan_batch_src_node_embeddings = batch_src_node_embeddings
        self.ctan_batch_dst_node_embeddings = batch_dst_node_embeddings
        
        if batch_neg_dst is None:
            batch_neg_src_node_embeddings = None
            batch_neg_dst_node_embeddings = None
        else:
            batch_neg_src = batch_neg[0]
            batch_neg_dst = batch_neg[1]
            batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = z[self.assoc[batch_neg_src]], z[self.assoc[batch_neg_dst]]

        
        return (batch_src_node_embeddings, batch_dst_node_embeddings), (batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings)