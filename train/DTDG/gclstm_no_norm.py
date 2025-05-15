import argparse
import os
from typing import Dict

from torch.nn.modules import Module

from ...model.gclstm_no_norm import MultiLayerGCLSTM_no_norm, GCLSTMParam
from .trainer import DTDGTrainer

class GCLSTMTrainer_no_norm(DTDGTrainer):
    def set_model_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--num-units', type=int, help='Number of GCLSTM units to stack', default=1)
        parser.add_argument('--out-channels', type=int, help='Output channel dimension of GCLSTM', default=100)
        parser.add_argument('--k-gclstm', type=int, help='Chebyshev filter size', default=1)
        
        return parser
    
    def get_model(self) -> Dict[str, Module]:
        models = super(GCLSTMTrainer_no_norm, self).get_model()

        gnn = MultiLayerGCLSTM_no_norm(
            num_units=self.args.num_units, 
            gclstm_param=GCLSTMParam(
                in_channels=self.train_loader.dataset._node_feat.size(1),
                out_channels=self.args.out_channels,
                K=self.args.k_gclstm,
                normalization=None))

        gnn.to(self.device)
        models['node_emb'] = gnn

        return models

    def _get_run_save_dir(self) -> str:
        dtdgtrainer_run_save_dir = super(GCLSTMTrainer_no_norm, self)._get_run_save_dir()
        return os.path.join(dtdgtrainer_run_save_dir,
                            "gclstm_no_norm")