import argparse
import os
from typing import Dict

import torch
from torch.nn.modules import Module
from torch_geometric.nn.models.tgn import LastNeighborLoader

from ...dataset.CTDG.torch_dataset.link_pred.node_feat_static import ContinuousTimeLinkPredNodeFeatureStaticDataset
from ...model.CTAN.ctan import CTAN
from .trainer import CTDGTrainer

class CTANTrainer(CTDGTrainer):
       
    def set_model_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--num-units', type=int, help='Number of CTAN units', default=1)
        parser.add_argument('--embedding-dim', type=int, help='Memory dimension of CTAN')
        parser.add_argument('--activation-layer', type=str, choices=['tanh'], help='Activation layer used within CTAN')
        parser.add_argument('--time-feat-dim', type=int, help='dimension of the time embedding')
        parser.add_argument('--epsilon', type=float)
        parser.add_argument('--gamma', type=float)
        parser.add_argument('--out-dim', type=float)
        parser.add_argument('--mean-delta-t', type=float)
        parser.add_argument('--std-delta-t', type=float)
        parser.add_argument('--init-time', type=float)
        parser.add_argument('--sampler-size', type=int)

        return parser

    
    def get_model(self) -> Dict[str, Module]:
        models = super(CTANTrainer, self).get_model()
        self.full_dataset = ContinuousTimeLinkPredNodeFeatureStaticDataset(os.path.join(self.args.root_load_save_dir, self.args.data_loc), 
                                                                      self.args.data, 
                                                                      "all", 
                                                                      self.args.node_feat, 
                                                                      self.args.node_feat_dim)
        self.neighbor_loader = LastNeighborLoader(self.full_dataset.num_nodes, size=self.args.sampler_size, device=self.device)
        self.assoc = torch.empty(self.full_dataset.num_nodes, dtype=torch.long, device=self.device)

        backbone = CTAN(num_nodes=self.full_dataset.num_nodes, 
                        edge_dim=self.full_dataset.edge_feat.shape[1], 
                        memory_dim=self.args.embedding_dim, 
                        time_dim=self.args.time_feat_dim,
                        node_dim=self.full_dataset._node_feat.shape[1], 
                        num_iters=self.args.num_units,
                        gnn_act=self.args.activation_layer,
                        gnn_act_kwargs=None,
                        epsilon=self.args.epsilon,
                        gamma=self.args.gamma,
                        mean_delta_t=self.args.mean_delta_t, 
                        std_delta_t=self.args.std_delta_t,
                        init_time=0,
                        conv_type='TransformerConv')
        backbone.to(self.device)
        models['node_emb'] = backbone

        return models

    def _get_run_save_dir(self) -> str:
        ctdgtrainer_run_save_dir = super(CTANTrainer, self)._get_run_save_dir()
        return os.path.join(ctdgtrainer_run_save_dir,
                            "ctan")