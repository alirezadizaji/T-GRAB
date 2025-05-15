from torch_geometric.nn.models import GCN as BaseGCN
from .node_emb import NodeEmbeddingModel

class GCN(BaseGCN, NodeEmbeddingModel):
    @property
    def out_dimension(self):
       return self.hidden_channels