import torch

from torch import nn
from torch.utils.data import DataLoader, Dataset
from utils.model_util import LeNet5
from utils.data_util import *
from utils.train_util import eval_model


x = torch.rand([1, 28, 28])
print(x.shape)
a = nn.Sequential(
    nn.Conv2d(1, 2, 5),
    nn.ReLU(inplace=True),
    nn.AvgPool2d(kernel_size=2))
b = nn.Sequential(
    nn.Conv2d(2, 2, 5),
    nn.ReLU(inplace=True),
    nn.AvgPool2d(kernel_size=2))

print(b(a(x)).shape)