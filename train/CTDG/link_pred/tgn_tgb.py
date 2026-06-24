from typing import Optional, Tuple

import torch
import numpy as np
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

        if self.use_dyglib:
            self.model[NODE_EMB_MODEL_NAME].train()
            self.model[NODE_EMB_MODEL_NAME].neighbor_sampler = self.train_neighbor_sampler

    def after_iteration_training(self):
        self.model['memory'].detach()

    def after_epoch_training(self):
        pass

    def before_epoch_evaluation(self, split_mode: str):
        if split_mode == 'train':
            self.before_epoch_training()
            self.model[NODE_EMB_MODEL_NAME].eval()
            self.model['memory'].eval()
        else:
            if self.use_dyglib:
                self.model[NODE_EMB_MODEL_NAME].neighbor_sampler = self.full_neighbor_sampler
                self.model[NODE_EMB_MODEL_NAME].eval()
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
        


        if self.use_dyglib:
            z, last_update = self.model['memory'](torch.arange(self.num_nodes, device=self.device))

            src_node_ids = batch_src.cpu().numpy()
            dst_node_ids = batch_dst.cpu().numpy()
            node_interact_times = batch_t.cpu().numpy()

            node_ids = np.concatenate([src_node_ids, dst_node_ids])
            node_embeddings = self.model[NODE_EMB_MODEL_NAME].compute_node_temporal_embeddings(node_memories=z,
                                                                                     node_ids=node_ids,
                                                                                     node_interact_times=np.concatenate([node_interact_times,
                                                                                                                         node_interact_times]),
                                                                                     current_layer_num=self.num_layers,
                                                                                     num_neighbors=self.args.num_neighbors)
            # two Tensors, with shape (batch_size, node_feat_dim)
            batch_src_node_embeddings, batch_dst_node_embeddings = node_embeddings[:len(src_node_ids)], node_embeddings[
                                                                                            len(src_node_ids): len(
                                                                                                src_node_ids) + len(
                                                                                                dst_node_ids)]


            if batch_neg is None:
                batch_neg_src_node_embeddings = None
                batch_neg_dst_node_embeddings = None
            else:
                neg_src_node_ids = batch_neg[0]
                neg_dst_node_ids = batch_neg[1]

                z_neg, last_update = self.model['memory'](torch.arange(self.num_nodes, device=self.device))

                neg_src_node_ids = neg_src_node_ids.cpu().numpy()
                neg_dst_node_ids = neg_dst_node_ids.cpu().numpy()
                neg_interact_times = batch_neg[2].cpu().numpy()
                neg_node_ids = np.concatenate([neg_src_node_ids, neg_dst_node_ids])

                negative_node_embeddings = self.model[NODE_EMB_MODEL_NAME].compute_node_temporal_embeddings(node_memories=z_neg,
                                                                                         node_ids=neg_node_ids,
                                                                                         node_interact_times=np.concatenate(
                                                                                             [neg_interact_times,
                                                                                              neg_interact_times]),
                                                                                         current_layer_num=self.num_layers,
                                                                                         num_neighbors=self.args.num_neighbors)
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = negative_node_embeddings[
                                                                       :len(neg_src_node_ids)], negative_node_embeddings[
                                                                                            len(neg_src_node_ids): len(
                                                                                                neg_src_node_ids) + len(
                                                                                                neg_dst_node_ids)]

        else:
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

            if batch_neg is None:
                batch_neg_src_node_embeddings = None
                batch_neg_dst_node_embeddings = None
            else:
                batch_neg_src = batch_neg[0]
                batch_neg_dst = batch_neg[1]
                batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings = z[self.assoc[batch_neg_src]], z[
                    self.assoc[batch_neg_dst]]


        self.model['memory'].update_state(batch_src, batch_dst, batch_t, batch_edge_feat.float())
        self.neighbor_loader.insert(
            batch_src,
            batch_dst
        )

        return (batch_src_node_embeddings, batch_dst_node_embeddings), (batch_neg_src_node_embeddings, batch_neg_dst_node_embeddings)