import argparse
import os
from typing import Dict

from torch.nn.modules import Module

from ...model.evolve_gcnh_batch_separate import MultiLayerEGCNH, EvolveGCNParams
from .trainer import DTDGTrainer

class EvolveGCNHTrainer(DTDGTrainer):
    def set_model_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--num-units', type=int, help='Number of EvolveGCNO units', default=1)
        parser.add_argument('--in-channels', type=int, help='input channel dimension of EvolveGCNO', default=100)
        parser.add_argument('--improved', type=bool, help='If True, then identity is added to adjacency matrix', default=False)
        parser.add_argument('--cached', type=bool, help='If True, then EvolveGCN caches the normalized adjacency matrix and uses it in next steps', default=False)
        parser.add_argument('--normalize', type=bool, help='If True, then EvolveGCN normalizes the adjacency matrix', default=True)
        
        return parser
    
    def get_model(self) -> Dict[str, Module]:
        models = super(EvolveGCNHTrainer, self).get_model()
        gnn = MultiLayerEGCNH(
            num_units=self.args.num_units, 
            base_args=EvolveGCNParams(
                self.train_loader.dataset.num_nodes,
                self.args.in_channels,
                self.args.improved,
                self.args.cached,
                self.args.normalize),
                inp_dim=self.train_loader.dataset._node_feat.size(1))

        gnn.to(self.device)
        models['node_emb'] = gnn

        return models

    def _get_run_save_dir(self) -> str:
        dtdgtrainer_run_save_dir = super(EvolveGCNHTrainer, self)._get_run_save_dir()
        return os.path.join(dtdgtrainer_run_save_dir,
                            "egcnh_batch_sep")