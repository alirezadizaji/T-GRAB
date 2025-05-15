import torch
from torch import nn
from torch.nn import functional as F
class LinkPredictor(nn.Module):
    def __init__(self, hidden_dim: int, num_layers: int, p_dropout: float=0.2) -> None:
        super(LinkPredictor, self).__init__()

        ml = nn.ModuleList([])
        for i in range(num_layers):
            h = hidden_dim
            if i == 0:
                h = h * 2
            ml.append(nn.ReLU())
            ml.append(nn.Linear(h, hidden_dim))
        
        ml.append(nn.Dropout(p_dropout))
        ml.append(nn.Linear(hidden_dim, 1))
        self.seq = nn.Sequential(*ml)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = torch.cat([x1, x2], dim=-1)
        x = self.seq(x)
        x = F.sigmoid(x)

        return x
    