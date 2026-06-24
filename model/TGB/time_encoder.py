"""
Time Encoding Module

Reference:
    - https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/nn/models/tgn.html
"""


import torch
from torch import Tensor
from torch.nn import Linear


class TimeEncoder(torch.nn.Module):
    def __init__(self, out_channels: int):
        super().__init__()
        self.out_channels = out_channels
        self.lin = Linear(1, out_channels)

    def reset_parameters(self):
        self.lin.reset_parameters()

    def forward(self, t: Tensor) -> Tensor:
        # Really bad hack. But necessary to make DyGLib and TGB compatible.
        if t.dim() == 2:
            # Tensor, shape (batch_size, seq_len, 1)
            timestamps = t.unsqueeze(dim=2)

            # Tensor, shape (batch_size, seq_len, time_dim)
            output = torch.cos(self.lin(timestamps))
            return output
        else:
            return self.lin(t.view(-1, 1)).cos()