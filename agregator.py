import torch
from copy import deepcopy



class Aggregator:
    def __init__(self, args, args_train):
        if args.algorithm == 0 or  args.algorithm == 1:
            self.aggregator = aggregate

    def aggregate(self):
        return self