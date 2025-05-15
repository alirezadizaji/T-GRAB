import argparse
import os
from typing import Dict

import torch
from torch.nn.modules import Module

from ...dataset.CTDG.torch_dataset.link_pred.node_feat_static import ContinuousTimeLinkPredNodeFeatureStaticDataset
from ...model.TGB import GraphAttentionEmbedding, TGNMemory, LastAggregator, IdentityMessage, LastNeighborLoader
from .trainer import CTDGTrainer

class TGNTrainer(CTDGTrainer):
       
    def set_model_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--num-units', type=int, help='Number of TGN units', default=1)
        parser.add_argument('--num-heads', type=int, help='Number of attention heads in TGN', default=2)
        parser.add_argument('--dropout', type=float, help='TGN: dropout rate', default=0.1)
        parser.add_argument('--num-neighbors', type=int, help='TGN: number of neighbors to sample for each node', default=20)
        parser.add_argument('--time-feat-dim', type=int, default=100, help='dimension of the time embedding')
        parser.add_argument('--memory-dim', type=int, default=100, help='dimension of the memory')
        return parser

    
    def get_model(self) -> Dict[str, Module]:
        models = super(TGNTrainer, self).get_model()
        self.full_dataset = ContinuousTimeLinkPredNodeFeatureStaticDataset(os.path.join(self.args.root_load_save_dir, self.args.data_loc), 
                                                                      self.args.data, 
                                                                      "all", 
                                                                      self.args.node_feat, 
                                                                      self.args.node_feat_dim)

        self.edge_feats = self.full_dataset.edge_feat.to(self.device)
        self.t = self.full_dataset.t.to(self.device)

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

        backbone = GraphAttentionEmbedding(
                in_channels=self.full_dataset._node_feat.size(-1),
                out_channels=self.full_dataset._node_feat.size(-1),
                msg_dim=self.full_dataset.edge_feat.size(-1),
                time_enc=memory.time_enc).to(self.device)

        self.neighbor_loader = LastNeighborLoader(self.full_dataset.num_nodes, size=self.args.num_neighbors, device=self.device)
        self.assoc = torch.empty(self.full_dataset.num_nodes, dtype=torch.long, device=self.device)

        
        models['memory'] = memory
        models['node_emb'] = backbone

        return models

    def _get_run_save_dir(self) -> str:
        ctdgtrainer_run_save_dir = super(TGNTrainer, self)._get_run_save_dir()
        return os.path.join(ctdgtrainer_run_save_dir,
                            "tgn_tgb")