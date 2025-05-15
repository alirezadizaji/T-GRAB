import argparse
import os
from typing import Dict

from torch.nn.modules import Module

# from ...model.evolve_gcno_custom import MultiLayerEGCNO, EvolveGCNParams
from ...model.IBM.egcno import EGCN as EvolveGCNO, EvolveGCNParams

from .trainer import DTDGTrainer

class EvolveGCNTrainer(DTDGTrainer):
    def set_model_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--num-units', type=int, help='Number of EvolveGCNO units', default=1)
        parser.add_argument('--out-channels', type=int, help='input channel dimension of EvolveGCNO', default=100)
        parser.add_argument('--improved', type=bool, help='If True, then identity is added to adjacency matrix', default=False)
        parser.add_argument('--cached', type=bool, help='If True, then EvolveGCN caches the normalized adjacency matrix and uses it in next steps', default=False)
        parser.add_argument('--normalize', type=bool, help='If True, then EvolveGCN normalizes the adjacency matrix', default=True)
        
        return parser
    
    def get_model(self) -> Dict[str, Module]:
        models = super(EvolveGCNTrainer, self).get_model()
        in_channels = self.train_loader.dataset._node_feat.size(1)

        from torch.nn import RReLU

        gnn = EvolveGCNO(
            params=EvolveGCNParams(
                    in_channels,
                    self.args.out_channels,
                    activation=RReLU()),
            num_layers=self.args.num_units,
            device=self.device)

        models['node_emb'] = gnn

        return models

    def _get_run_save_dir(self) -> str:
        dtdgtrainer_run_save_dir = super(EvolveGCNTrainer, self)._get_run_save_dir()
        return os.path.join(dtdgtrainer_run_save_dir,
                            "egcno")