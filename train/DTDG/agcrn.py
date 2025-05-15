import argparse
import os
from typing import Dict

from torch.nn.modules import Module
 
from ...model.agcrn import MultiLayerAGCRN, AGCRNParams
from .trainer import DTDGTrainer

class AGCRNTrainer(DTDGTrainer):
    def set_model_args(self, parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        parser.add_argument('--num-units', type=int, help='Number of AGCRN units', default=1)
        parser.add_argument('--out-channels', type=int, help='Output channel dimension of AGCRN', default=100)
        parser.add_argument('--k-agcrn', type=int, help='AGCRN filter size', default=1)
        
        return parser

    
    def get_model(self) -> Dict[str, Module]:
        models = super(AGCRNTrainer, self).get_model()
        gnn = MultiLayerAGCRN(
            num_units=self.args.num_units, 
            base_args=AGCRNParams(
                number_of_nodes=self.train_loader.dataset.num_nodes,
                embedding_dimensions=self.train_loader.dataset._node_feat.shape[1],
                in_channels=self.train_loader.dataset._node_feat.shape[1],
                out_channels=self.args.out_channels,
                K=self.args.k_agcrn))

        gnn.to(self.device)
        models['node_emb'] = gnn

        return models

    def _get_run_save_dir(self) -> str:
        dtdgtrainer_run_save_dir = super(AGCRNTrainer, self)._get_run_save_dir()
        return os.path.join(dtdgtrainer_run_save_dir,
                            "agcrn")