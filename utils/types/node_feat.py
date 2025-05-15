class NodeFeatType:
    CONSTANT = "CONSTANT"
    RAND = "RAND" # Sampling from uniform dist
    RANDN = "RANDN" # Sampling from normal dist
    ONE_HOT = "ONE_HOT"
    NODE_ID = "NODE_ID"

    @staticmethod
    def list():
        return [
            NodeFeatType.CONSTANT, 
            NodeFeatType.RAND,
            NodeFeatType.RANDN, 
            NodeFeatType.ONE_HOT, 
            NodeFeatType.NODE_ID
        ]