import torch

from ...DTDG.link_pred.trainer import LinkPredTrainer
from ...DTDG.agcrn import AGCRNTrainer
from ...DTDG.trainer import NODE_EMB_MODEL_NAME

class LinkPredAGCRNTrainer(LinkPredTrainer, AGCRNTrainer):
    def __init__(self):
        super().__init__()
        
        self.E = torch.nn.Parameter(
            torch.randn(self.train_loader.dataset.num_nodes, self.train_loader.dataset._node_feat.shape[1]), requires_grad=True
        ).to(self.device)

    def before_training(self):
        pass

    def forward_backbone(self, snapshot: torch.Tensor, node_feat: torch.Tensor) -> torch.Tensor:
        z = self.model[NODE_EMB_MODEL_NAME](
                node_feat.unsqueeze(0),
                self.E).squeeze(0)
        
        return z