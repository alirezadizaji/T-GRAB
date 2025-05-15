import torch

from torch_geometric.nn.models.tgn import IdentityMessage, LastAggregator, TimeEncoder
from torch_geometric.nn.resolver import activation_resolver
from typing import Callable, Optional, Any, Dict, Union, List
from .predictors import *
from .memory_layers import *
from .gnn_layers import *
from ..node_emb import NodeEmbeddingModel
import numpy as np


class GenericModel(torch.nn.Module):
    
    def __init__(self, num_nodes, memory=None, gnn=None, gnn_act=None):
        super(GenericModel, self).__init__()
        self.memory = memory
        self.gnn = gnn
        self.gnn_act = gnn_act
        self.num_gnn_layers = 1
        self.num_nodes = num_nodes

    def reset_memory(self):
        if self.memory is not None: self.memory.reset_state()

    def zero_grad_memory(self):
        if self.memory is not None: self.memory.zero_grad_memory()

    def update(self, src, pos_dst, t, msg, *args, **kwargs):
        if self.memory is not None: self.memory.update_state(src, pos_dst, t, msg)

    def detach_memory(self):
        if self.memory is not None: self.memory.detach()
    
    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        super().reset_parameters()
        if hasattr(self.memory, 'reset_parameters'):
            self.memory.reset_parameters()
        if hasattr(self.gnn, 'reset_parameters'):
                    self.gnn.reset_parameters()
    

class CTAN(GenericModel, NodeEmbeddingModel):
    def __init__(self, 
                 # Memory params
                 num_nodes: int, 
                 edge_dim: int, 
                 memory_dim: int, 
                 time_dim: int,
                 node_dim: int = 0, 
                 # CTAN params
                 num_iters: int = 1,
                 gnn_act: Union[str, Callable, None] = 'tanh',
                 gnn_act_kwargs: Optional[Dict[str, Any]] = None,
                 epsilon: float = 0.1,
                 gamma: float = 0.1,
                 # Mean and std values for normalization
                 mean_delta_t: float = 0., 
                 std_delta_t: float = 1.,
                 init_time: int = 0,
                 # conv type Transformer
                 conv_type: str = 'TransformerConv'
        ):

        assert conv_type in ['TransformerConv']

        # Define memory
        memory = SimpleMemory(
            num_nodes,
            memory_dim,
            aggregator_module=LastAggregator(),
            init_time=init_time
        )

        gnn = CTANEmbedding(memory_dim, node_dim, edge_dim, time_dim, num_iters, gnn_act, gnn_act_kwargs, 
                            epsilon, gamma, mean_delta_t,  std_delta_t, conv_type=conv_type)
        
        # Original implementation puts backbone and link predictor together in one module.
        # This framework implements a shared link predictor among all types of backbone models.
        # So there is no need to keep link predictor here.
        readout = None
        super().__init__(num_nodes, memory, gnn, gnn_act)

        self.num_gnn_layers = num_iters

    def zero_grad_memory(self):
        if self.memory is not None: self.memory.zero_grad_memory()
        
    def update(self, src, pos_dst, t, msg, src_emb, pos_dst_emb):
        self.memory.update_state(src, pos_dst, t, src_emb, pos_dst_emb)

    @property
    def out_dimension(self):
        return self.gnn.out_channels
        
    def forward(self, x, n_id, msg, t, edge_index):
        # Get updated memory of all nodes involved in the computation.
        z, last_update = self.memory(n_id)

        if len(x.shape) == 3: # sequence classification case
            x = x.squeeze(0)
        elif len(x.shape) == 2: # link-based predictions
            pass
        else:
            raise ValueError(f"Unexpected node feature shape. Got {x.shape}")
        z = torch.cat((z, x[n_id]), dim=-1)

        z = self.gnn(z, last_update, edge_index, t, msg)
        
        return z