from dataclasses import dataclass

import torch
from torch import nn
from torch_geometric_temporal.nn.recurrent.evolvegcno import GCNConv_Fixed_W, GRU, EvolveGCNO as OriginalEGCNO

from .node_emb import NodeEmbeddingModel

@dataclass
class EvolveGCNParams:
    in_channels: int
    out_channels: int
    improved: bool = False
    cached: bool = False
    normalize: bool = True


class EvolveGCNO(OriginalEGCNO):
    """ An extension implementation of original EvolveGCNO from torch_geometric_temporal to support RNN forwarding with different hidden dimensions between X and self.initial_weight """
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        improved: bool = False,
        cached: bool = False,
        normalize: bool = True,
        add_self_loops: bool = True,
    ):
        self.out_channels = out_channels
        super(EvolveGCNO, self).__init__(
            in_channels,
            improved,
            cached,
            normalize,
            add_self_loops)

        del self.initial_weight
        self.initial_weight = torch.nn.Parameter(torch.Tensor(1, in_channels, out_channels))
        self.reset_parameters()

    def _create_layers(self):
        super(EvolveGCNO, self)._create_layers()
        del self.recurrent_layer
        self.recurrent_layer = GRU(
            input_size=self.out_channels, hidden_size=self.out_channels, num_layers=1
        )
        for param in self.recurrent_layer.parameters():
            param.requires_grad = True
            param.retain_grad()
        
        del self.conv_layer
        self.conv_layer = GCNConv_Fixed_W(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            improved=self.improved,
            cached=self.cached,
            normalize=self.normalize,
            add_self_loops=self.add_self_loops
        )

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
    
    def forward(self, X: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor) -> torch.Tensor:
        for unit in self.units:
            X = unit(X, edge_index, edge_weight=edge_weight)
        
        return X