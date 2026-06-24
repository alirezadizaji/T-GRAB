import argparse
import os
from typing import Dict

import torch
from torch.nn.modules import Module

from ...dataset.CTDG.torch_dataset.link_pred.node_feat_static import ContinuousTimeLinkPredNodeFeatureStaticDataset
from ...model.TGB import GraphAttentionEmbedding, TGNMemory, LastAggregator, IdentityMessage, LastNeighborLoader
from .trainer import CTDGTrainer
from ...model.DyGLib.memory_model import GraphAttentionEmbedding as dyglib_GraphAttentionEmbedding
from ...model.DyGLib.utils import get_neighbor_sampler
class TGNTrainer(CTDGTrainer):
       
    def set_model_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--num-units', type=int, help='Number of TGN units', default=1)
        parser.add_argument('--num-heads', type=int, help='Number of attention heads in TGN', default=2)
        parser.add_argument('--dropout', type=float, help='TGN: dropout rate', default=0.1)
        parser.add_argument('--num-neighbors', type=int, help='TGN: number of neighbors to sample for each node', default=20)
        parser.add_argument('--time-feat-dim', type=int, default=100, help='dimension of the time embedding')
        parser.add_argument('--memory-dim', type=int, default=100, help='dimension of the memory')
        parser.add_argument('--use-dyglib', action='store_true', help='use dyglib for embedding')
        return parser

    
    def get_model(self) -> Dict[str, Module]:
        models = super(TGNTrainer, self).get_model()
        self.use_dyglib = self.args.use_dyglib
        if self.use_dyglib:
            print("Using DyGLib for embedding")

        self.num_layers = self.args.num_units
        print("Using number of heads {}".format(self.args.num_heads))

        self.full_dataset = ContinuousTimeLinkPredNodeFeatureStaticDataset(os.path.join(self.args.root_load_save_dir, self.args.data_loc), 
                                                                      self.args.data, 
                                                                      "all", 
                                                                      self.args.node_feat, 
                                                                      self.args.node_feat_dim)

        self.train_neighbor_sampler = get_neighbor_sampler(src_node_ids=self.train_loader.dataset.src,
                                                           dst_node_ids=self.train_loader.dataset.dst,
                                                           edge_ids=self.train_loader.dataset.edge_ids,
                                                           node_interact_times=self.train_loader.dataset.t,
                                                           sample_neighbor_strategy='recent',
                                                           time_scaling_factor=1e-6,
                                                           seed=0)

        self.full_neighbor_sampler = get_neighbor_sampler(src_node_ids=self.full_dataset.src,
                                                    dst_node_ids=self.full_dataset.dst,
                                                    edge_ids=self.full_dataset.edge_ids,
                                                    node_interact_times=self.full_dataset.t,
                                                    sample_neighbor_strategy='recent',
                                                    time_scaling_factor=1e-6,
                                                    seed=0)


        self.edge_feats = self.full_dataset.edge_feat.to(self.device)
        self.t = self.full_dataset.t.to(self.device)
        self.num_nodes = self.full_dataset.num_nodes

        memory = TGNMemory(
                    self.full_dataset.num_nodes,
                    self.full_dataset.edge_feat.size(-1),
                    self.full_dataset._node_feat.size(-1),
                    self.args.time_feat_dim,
                    self.full_dataset._node_feat.to(self.device),
                    message_module=IdentityMessage(self.full_dataset.edge_feat.size(-1), 
                                                   self.full_dataset._node_feat.size(-1), 
                                                   self.args.time_feat_dim),
                    aggregator_module=LastAggregator()).to(self.device)

        if self.use_dyglib:
            dyglib_embedding = dyglib_GraphAttentionEmbedding(node_raw_features=self.full_dataset._node_feat.to(self.device),
                                                                edge_raw_features=self.full_dataset.edge_feat.to(self.device),
                                                                neighbor_sampler=self.train_neighbor_sampler,
                                                                time_encoder=memory.time_enc,
                                                                node_feat_dim=self.full_dataset._node_feat.size(-1),
                                                                edge_feat_dim=self.full_dataset.edge_feat.size(-1),
                                                                time_feat_dim=self.args.time_feat_dim,
                                                                num_layers=self.args.num_units,
                                                                num_heads=self.args.num_heads,
                                                                dropout=self.args.dropout)
            dyglib_embedding.out_dimension = self.full_dataset._node_feat.size(-1)
            dyglib_embedding.to(self.device)
            models['node_emb'] = dyglib_embedding

        else:
            backbone = GraphAttentionEmbedding(
                    in_channels=self.full_dataset._node_feat.size(-1),
                    out_channels=self.full_dataset._node_feat.size(-1),
                    msg_dim=self.full_dataset.edge_feat.size(-1),
                    time_enc=memory.time_enc).to(self.device)
            models['node_emb'] = backbone

        self.neighbor_loader = LastNeighborLoader(self.full_dataset.num_nodes, size=self.args.num_neighbors, device=self.device)
        self.assoc = torch.empty(self.full_dataset.num_nodes, dtype=torch.long, device=self.device)
        
        models['memory'] = memory
        return models

    def _get_run_save_dir(self) -> str:
        ctdgtrainer_run_save_dir = super(TGNTrainer, self)._get_run_save_dir()
        return os.path.join(ctdgtrainer_run_save_dir,
                            "tgn_tgb")