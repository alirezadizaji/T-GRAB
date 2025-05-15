from ....DTDG.link_pred.gcn import LinkPredGCNTrainer
from ....DTDG.link_pred.long_range.trainer import LongRangeTrainer

class LongRangeGCNTrainer(LongRangeTrainer, LinkPredGCNTrainer):
    pass