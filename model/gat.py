from torch_geometric.nn.models import GAT as BaseGAT
from .node_emb import NodeEmbeddingModel

class GAT(BaseGAT, NodeEmbeddingModel):
    @property
    def out_dimension(self):
       return self.hidden_channels