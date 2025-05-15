from dataclasses import dataclass, asdict

import torch
from torch import nn
from torch_geometric_temporal import TGCN

from .node_emb import NodeEmbeddingModel

@dataclass
class TGCNParams:
    in_channels: int
    out_channels: int
    improved: bool = False
    cached: bool = False

class MultiLayerTGCN(NodeEmbeddingModel):
    def __init__(self, num_units: int, base_args: TGCNParams) -> None:
        super(MultiLayerTGCN, self).__init__()
        self.num_units = num_units
        self.base_args = base_args
        self.units: nn.ModuleList[TGCN] = nn.ModuleList([])
        self.units.append( 
                TGCN(in_channels=self.base_args.in_channels, 
                     out_channels=self.base_args.out_channels, 
                     improved=self.base_args.improved, 
                     cached=self.base_args.cached, 
                     add_self_loops=False)
                )
        
        for _ in range(1, num_units):
            self.units.append( 
                TGCN(in_channels=self.base_args.out_channels, 
                     out_channels=self.base_args.out_channels, 
                     improved=self.base_args.improved, 
                     cached=self.base_args.cached, 
                     add_self_loops=False)
                )
        
    @property
    def out_dimension(self):
        return self.base_args.out_channels
    
    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, hs: torch.Tensor, edge_weight: torch.Tensor = None) -> torch.Tensor:
        inp = X
        out_hs = []

        for i in range(self.num_units):
            h = self.units[i](inp, edge_index, edge_weight=edge_weight, H=hs[i])
            out_hs.append(h)
            inp = h
        
        h1 = torch.stack(out_hs)
        return h1, h