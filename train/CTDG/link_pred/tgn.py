from typing import Optional, Tuple

import torch

from ...CTDG.link_pred.trainer import LinkPredTrainer
from ...CTDG.tgn import TGNTrainer
from ...CTDG.trainer import NODE_EMB_MODEL_NAME

class LinkPredTGNTrainer(LinkPredTrainer, TGNTrainer):
    def __init__(self):
        super().__init__()
    
    def before_epoch_training(self): 
        self.model[NODE_EMB_MODEL_NAME].set_neighbor_sampler(self.train_neighbor_sampler)
        self.model[NODE_EMB_MODEL_NAME].memory_bank.__init_memory_bank__()

    def after_iteration_training(self):
        self.model[NODE_EMB_MODEL_NAME].memory_bank.detach_memory_bank()

    def after_epoch_training(self):
        pass

    def before_epoch_evaluation(self, split_mode: str):
        if split_mode == 'train':
            self.before_epoch_training()
        else:
            self.model[NODE_EMB_MODEL_NAME].set_neighbor_sampler(self.full_neighbor_sampler)
        # if not self.args.eval_mode:
        #     pass
        # else:
        #     for node_id, node_raw_messages in self.model[NODE_EMB_MODEL_NAME].memory_bank.node_raw_messages.items():
        #         new_node_raw_messages = []
        #         for node_raw_message in node_raw_messages:
        #             new_node_raw_messages.append((node_raw_message[0].to(self.device), node_raw_message[1]))
        #         self.model[NODE_EMB_MODEL_NAME].memory_bank.node_raw_messages[node_id] = new_node_raw_messages
            
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
        
        if batch_edge_id is not None:
            batch_edge_id = batch_edge_id.cpu().numpy()
    
        batch_src_node_embeddings, batch_dst_node_embeddings = \
        self.model[NODE_EMB_MODEL_NAME].compute_src_dst_node_temporal_embeddings(
                                                        src_node_ids=batch_src.cpu().numpy(),
                                                        dst_node_ids=batch_dst.cpu().numpy(),
                                                        node_interact_times=batch_t.cpu().numpy(),
                                                        edge_ids=batch_edge_id,
                                                        edges_are_positive=True,
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
                                                        edge_ids=None,
                                                        edges_are_positive=False,
                                                        num_neighbors=self.args.num_neighbors)

        return (batch_src_node_embeddings, batch_dst_node_embeddings), (batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings)