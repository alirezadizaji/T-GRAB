import argparse
import os
from typing import Dict

from torch.nn.modules import Module

from ...model.evolve_gcno_lstm import MultiLayerEGCNO, EvolveGCNParams
from .trainer import DTDGTrainer

class EvolveGCNLSTMTrainer(DTDGTrainer):
    def set_model_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--num-units', type=int, help='Number of EvolveGCNO units', default=1)
        parser.add_argument('--in-channels', type=int, help='input channel dimension of EvolveGCNO', default=100)
        parser.add_argument('--improved', type=bool, help='If True, then identity is added to adjacency matrix', default=False)
        parser.add_argument('--cached', type=bool, help='If True, then EvolveGCN caches the normalized adjacency matrix and uses it in next steps', default=False)
        parser.add_argument('--normalize', type=bool, help='If True, then EvolveGCN normalizes the adjacency matrix', default=True)
        parser.add_argument('--lstm_forget_scale', type=float, default=1.0)
        parser.add_argument('--lstm_input_scale', type=float, default=1.0)
        return parser
    
    def get_model(self) -> Dict[str, Module]:
        models = super(EvolveGCNLSTMTrainer, self).get_model()
        gnn = MultiLayerEGCNO(
            num_units=self.args.num_units, 
            base_args=EvolveGCNParams(
                self.args.in_channels,
                self.args.improved,
                self.args.cached,
                self.args.normalize,
                self.args.lstm_input_scale,
                self.args.lstm_forget_scale),
                inp_dim=self.train_loader.dataset._node_feat.size(1))

        gnn.to(self.device)
        models['node_emb'] = gnn

        return models

    def _get_run_save_dir(self) -> str:
        dtdgtrainer_run_save_dir = super(EvolveGCNLSTMTrainer, self)._get_run_save_dir()
        return os.path.join(dtdgtrainer_run_save_dir,
                            "egcno_lstm")