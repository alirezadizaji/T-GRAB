from typing import Any

import torch

from ..utils.types import NodeFeatType

class NodeFeatGenerator:
    def __init__(self, type: NodeFeatType):
        self.type = type
    
    def __call__(self, num_nodes: int, node_feat_dim: int = 1) -> torch.Tensor:
        node_feat = None

        if self.type == NodeFeatType.CONSTANT:
            node_feat = torch.ones((num_nodes, node_feat_dim), dtype=torch.float32)
        elif self.type == NodeFeatType.RAND:
            node_feat = torch.rand((num_nodes, node_feat_dim), dtype=torch.float32)
        elif self.type == NodeFeatType.RANDN:
            node_feat = torch.randn((num_nodes, node_feat_dim), dtype=torch.float32)
        elif self.type == NodeFeatType.ONE_HOT:
            node_feat = torch.eye(num_nodes).float()
        elif self.type == NodeFeatType.NODE_ID:
            node_feat = torch.arange(num_nodes).unsqueeze(1).float()
        
        print(f"===========> Node feature generated: size {node_feat.size()}", flush=True)
        return node_feat