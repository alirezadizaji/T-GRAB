from typing import Optional, Tuple

import torch

from ...CTDG.link_pred.trainer import LinkPredTrainer
from ...CTDG.tgat import TGATTrainer
from ...CTDG.trainer import NODE_EMB_MODEL_NAME
from ...CTDG.link_pred.multitask_trainer import MultiTaskLinkPredTrainer

class LinkPredTGATMultiTaskTrainer(MultiTaskLinkPredTrainer, TGATTrainer):
    def __init__(self):
        super().__init__()
    
    def before_epoch_training(self):        
        self.model[NODE_EMB_MODEL_NAME].set_neighbor_sampler(self.train_neighbor_sampler)

    def before_iteration_training(self):
        pass

    def after_iteration_training(self):
        pass

    def after_epoch_training(self):
        pass

    def before_epoch_evaluation(self, split_mode):
        self.model[NODE_EMB_MODEL_NAME].set_neighbor_sampler(self.full_neighbor_sampler)
    
    def after_iteration_evaluation(self, split_mode):
        pass

    def after_epoch_evaluation(self, split_mode):
        pass

    def forward_backbone(self, 
                         batch_src: torch.Tensor, 
                         batch_dst: torch.Tensor, 
                         batch_t: torch.Tensor, 
                         batch_edge_id: torch.Tensor, 
                         batch_edge_feat: torch.Tensor,
                         batch_neg: Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]=None) -> Tuple[torch.Tensor, torch.Tensor]:
        
        batch_src_node_embeddings, batch_dst_node_embeddings = \
                self.model[NODE_EMB_MODEL_NAME].compute_src_dst_node_temporal_embeddings(
                                                        src_node_ids=batch_src.cpu().numpy(),
                                                        dst_node_ids=batch_dst.cpu().numpy(),
                                                        node_interact_times=batch_t.cpu().numpy(),
                                                        num_neighbors=self.args.num_neighbors)

        if batch_neg is None:
            batch_neg_src_node_embeddings = None
            batch_neg_dst_node_embeddings = None
        else:
            batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = \
                self.model[NODE_EMB_MODEL_NAME].compute_src_dst_node_temporal_embeddings(
                                                        src_node_ids=batch_neg[0].cpu().numpy(),
                                                        dst_node_ids=batch_neg[1].cpu().numpy(),
                                                        node_interact_times=batch_neg[2].cpu().numpy(),
                                                        num_neighbors=self.args.num_neighbors)

        return (batch_src_node_embeddings, batch_dst_node_embeddings), (batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings)