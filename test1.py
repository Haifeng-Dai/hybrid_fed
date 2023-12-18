import torch

from torch.utils.data import DataLoader, Dataset
from utils.model_util import LeNet5
from utils.data_util import *
from utils.train_util import eval_model


a = [1, 2]
b = [a, 3]

b[0] = 'te'
print(a, b)
print(id(a))
print(id(b[0]))
