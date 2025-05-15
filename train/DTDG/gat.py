import argparse
import os
from typing import Dict

from torch.nn.modules import Module
from ...model.gat import GAT

from .trainer import DTDGTrainer

class GATTrainer(DTDGTrainer):
    def set_model_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--num-units', type=int, help='Number of GAT units', default=1)
        parser.add_argument('--out-channels', type=int, help='input channel dimension of GAT', default=100)
            
        return parser
    
    def get_model(self) -> Dict[str, Module]:
        models = super(GATTrainer, self).get_model()
        gnn = GAT(
                in_channels=self.train_loader.dataset._node_feat.size(-1),
                hidden_channels=self.args.out_channels,
                num_layers=self.args.num_units)
        
        gnn.to(self.device)
        models['node_emb'] = gnn

        return models

    def _get_run_save_dir(self) -> str:
        dtdgtrainer_run_save_dir = super(GATTrainer, self)._get_run_save_dir()
        return os.path.join(dtdgtrainer_run_save_dir,
                            "gat")
