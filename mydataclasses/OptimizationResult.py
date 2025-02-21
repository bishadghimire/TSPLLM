
class OptimizationResult(object):
    def __init__(self, *args) -> None:
        self.OrigNodeList = args[0]
        self.OptimizedNodeList = args[1]
        self.TourCost = args[2]




