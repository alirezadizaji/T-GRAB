from ....DTDG.link_pred.gclstm_no_norm import LinkPredGCLSTMTrainer_no_norm
from ....DTDG.link_pred.memory_node.trainer import MemoryNodeTrainer

class MemoryNodeGCLSTMTrainer(MemoryNodeTrainer, LinkPredGCLSTMTrainer_no_norm):
    pass