import argparse
import os
from typing import Dict

from torch.nn.modules import Module

from ...dataset.CTDG.torch_dataset.link_pred.node_feat_static import ContinuousTimeLinkPredNodeFeatureStaticDataset
from ...model.DyGLib.DyGFormer.DyGFormer import DyGFormer
from ...model.DyGLib.utils import get_neighbor_sampler
from .trainer import CTDGTrainer

class DyGFormerTrainer(CTDGTrainer):
       
    def set_model_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--num-units', type=int, help='Number of TGAT units', default=1)
        parser.add_argument('--num-heads', type=int, help='Number of attention heads in TGAT', default=2)
        parser.add_argument('--dropout', type=float, help='TGAT: dropout rate', default=0.1)
        parser.add_argument('--sample-neighbor-strategy', type=str, default='recent', choices=['uniform', 'recent', 'time_interval_aware'], help='how to sample historical neighbors')
        parser.add_argument('--time-scaling-factor', default=1e-6, type=float, help='the hyperparameter that controls the sampling preference with time interval, '
                        'a large time_scaling_factor tends to sample more on recent links, 0.0 corresponds to uniform sampling, '
                        'it works when sample_neighbor_strategy == time_interval_aware')
        parser.add_argument('--time-feat-dim', type=int, default=100, help='dimension of the time embedding')
        parser.add_argument('--patch_size', type=int, default=1, help='patch size')
        parser.add_argument('--channel_embedding_dim', type=int, default=50, help='dimension of each channel embedding')
        parser.add_argument('--max_input_sequence_length', type=int, default=32, help='maximal length of the input sequence of each node')

        return parser

    
    def get_model(self) -> Dict[str, Module]:
        models = super(DyGFormerTrainer, self).get_model()

        full_dataset = ContinuousTimeLinkPredNodeFeatureStaticDataset(os.path.join(self.args.root_load_save_dir, self.args.data_loc), 
                                                                      self.args.data, 
                                                                      "all", 
                                                                      self.args.node_feat, 
                                                                      self.args.node_feat_dim)

        self.train_neighbor_sampler = get_neighbor_sampler(src_node_ids=self.train_loader.dataset.src, 
                                                           dst_node_ids=self.train_loader.dataset.dst, 
                                                           edge_ids=self.train_loader.dataset.edge_ids,
                                                           node_interact_times=self.train_loader.dataset.t,
                                                           sample_neighbor_strategy=self.args.sample_neighbor_strategy,
                                                           time_scaling_factor=self.args.time_scaling_factor, 
                                                           seed=0)

        
        self.full_neighbor_sampler = get_neighbor_sampler(src_node_ids=full_dataset.src, 
                                                    dst_node_ids=full_dataset.dst, 
                                                    edge_ids=full_dataset.edge_ids,
                                                    node_interact_times=full_dataset.t,
                                                    sample_neighbor_strategy=self.args.sample_neighbor_strategy,
                                                    time_scaling_factor=self.args.time_scaling_factor, 
                                                    seed=0)       

        backbone = DyGFormer(node_raw_features=full_dataset._node_feat.numpy(), edge_raw_features=full_dataset.edge_feat.numpy(), neighbor_sampler=self.train_neighbor_sampler,
                                         time_feat_dim=self.args.time_feat_dim, channel_embedding_dim=self.args.channel_embedding_dim, patch_size=self.args.patch_size,
                                         num_layers=self.args.num_units, num_heads=self.args.num_heads, dropout=self.args.dropout,
                                         max_input_sequence_length=self.args.max_input_sequence_length, device=self.device)
        
        
        backbone.to(self.device)
        models['node_emb'] = backbone

        return models

    def _get_run_save_dir(self) -> str:
        ctdgtrainer_run_save_dir = super(DyGFormerTrainer, self)._get_run_save_dir()
        return os.path.join(ctdgtrainer_run_save_dir,
                            "dygformer")