from dataclasses import dataclass, asdict

import torch
from torch import nn
from torch_geometric_temporal.nn.recurrent.evolvegcno import GRU, EvolveGCNO

from .node_emb import NodeEmbeddingModel

@dataclass
class EvolveGCNParams:
    in_channels: int
    out_channels: int
    improved: bool = False
    cached: bool = False
    normalize: bool = True

class MultiLayerEGCNO(NodeEmbeddingModel):
    def __init__(self, num_units: int, base_args: EvolveGCNParams) -> None:
        super(MultiLayerEGCNO, self).__init__()

        self.base_args = base_args
        self.units: nn.ModuleList[EvolveGCNO] = nn.ModuleList([])
        self.units.append(
            EvolveGCNO(in_channels = self.base_args.in_channels,
                    out_channels = self.base_args.out_channels,
                    improved = self.base_args.improved,
                    cached = self.base_args.cached,
                    normalize = self.base_args.normalize,
                    add_self_loops=False)
        )

        for _ in range(1, num_units):
            self.units.append( 
                EvolveGCNO(in_channels = self.base_args.out_channels,
                    out_channels = self.base_args.out_channels,
                    improved = self.base_args.improved,
                    cached = self.base_args.cached,
                    normalize = self.base_args.normalize,
                    add_self_loops=False)
                )
        
    @property
    def out_dimension(self):
        return self.base_args.out_channels
    
    def forward(self, X: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        for unit in self.units:
            X = unit(X, edge_index)
        
        return X