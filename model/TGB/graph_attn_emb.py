"""
GNN-based modules used in the architecture of MP-TG models

"""

import math
from torch_geometric.nn import TransformerConv
import torch

from ..node_emb import NodeEmbeddingModel


class GraphAttentionEmbedding(NodeEmbeddingModel):
    """
    Reference:
    - https://github.com/pyg-team/pytorch_geometric/blob/master/examples/tgn.py
    """

    def __init__(self, in_channels, out_channels, msg_dim, time_enc):
        super().__init__()
        self.time_enc = time_enc
        edge_dim = msg_dim + time_enc.out_channels
        num_heads = 2
        self.out_channels = (out_channels) + (out_channels % num_heads)

        self.conv = TransformerConv(
            in_channels, self.out_channels // num_heads, heads=num_heads, dropout=0.1, edge_dim=edge_dim
        )

    @property
    def out_dimension(self) -> int:
        return self.out_channels

    def forward(self, x, last_update, edge_index, t, msg):
        rel_t = last_update[edge_index[0]] - t
        rel_t_enc = self.time_enc(rel_t.to(x.dtype))
        edge_attr = torch.cat([rel_t_enc, msg], dim=-1)
        res =  self.conv(x, edge_index, edge_attr)
        return res


class TimeEmbedding(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        class NormalLinear(torch.nn.Linear):
            # From TGN code: From JODIE code
            def reset_parameters(self):
                stdv = 1.0 / math.sqrt(self.weight.size(1))
                self.weight.data.normal_(0, stdv)
                if self.bias is not None:
                    self.bias.data.normal_(0, stdv)

        self.embedding_layer = NormalLinear(1, self.out_channels)

    def forward(self, x, last_update, t):
        rel_t = last_update - t
        embeddings = x * (1 + self.embedding_layer(rel_t.to(x.dtype).unsqueeze(1)))

        return embeddings