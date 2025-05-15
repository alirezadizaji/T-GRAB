import argparse
import os
from typing import Dict

from torch.nn.modules import Module

from ...model.tgcn import MultiLayerTGCN, TGCNParams
from .trainer import DTDGTrainer

class TGCNTrainer(DTDGTrainer):
    def set_model_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--num-units', type=int, help='Number of TGCN units', default=1)
        parser.add_argument('--out-channels', type=int, help='input channel dimension of TGCN', default=100)
        parser.add_argument('--improved', type=bool, help='If True, then identity is added to adjacency matrix', default=False)
        parser.add_argument('--cached', type=bool, help='If True, then TGCN caches the normalized adjacency matrix and uses it in next steps', default=False)
        
        return parser
    
    def get_model(self) -> Dict[str, Module]:
        models = super(TGCNTrainer, self).get_model()
        in_channels = self.train_loader.dataset._node_feat.size(1)
        gnn = MultiLayerTGCN(
            num_units=self.args.num_units, 
            base_args=TGCNParams(
                in_channels,
                self.args.out_channels,
                self.args.improved,
                self.args.cached))

        gnn.to(self.device)
        models['node_emb'] = gnn

        return models

    def _get_run_save_dir(self) -> str:
        dtdgtrainer_run_save_dir = super(TGCNTrainer, self)._get_run_save_dir()
        return os.path.join(dtdgtrainer_run_save_dir,
                            "tgcn")