from abc import abstractmethod

from torch import nn

class NodeEmbeddingModel(nn.Module):
    @property
    @abstractmethod
    def out_dimension(self):
        """ This returns the node embedding dimension after running the forward through the node-embedding model """
        pass