import argparse
import os
from typing import Dict

from torch.nn.modules import Module
from ...model.gcn import GCN

from .trainer import DTDGTrainer

class GCNTrainer(DTDGTrainer):
    def set_model_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--num-units', type=int, help='Number of GCN units', default=1)
        parser.add_argument('--out-channels', type=int, help='input channel dimension of GCN', default=100)
            
        return parser
    
    def get_model(self) -> Dict[str, Module]:
        models = super(GCNTrainer, self).get_model()
        gnn = GCN(
                in_channels=self.train_loader.dataset._node_feat.size(-1),
                hidden_channels=self.args.out_channels,
                num_layers=self.args.num_units)
        
        gnn.to(self.device)
        models['node_emb'] = gnn

        return models

    def _get_run_save_dir(self) -> str:
        dtdgtrainer_run_save_dir = super(GCNTrainer, self)._get_run_save_dir()
        return os.path.join(dtdgtrainer_run_save_dir,
                            "gcn")
