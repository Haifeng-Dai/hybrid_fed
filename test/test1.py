import torch

from torch import nn
from torch.utils.data import DataLoader, Dataset
from utils.model_util import LeNet5
from utils.data_util import *
from utils.train_util import eval_model


x = torch.rand([2, 1, 28, 28])
print(x.shape)
a = nn.Sequential(
    nn.Conv2d(1, 6, 5),
    nn.ReLU(inplace=True),
    nn.AvgPool2d(kernel_size=2))
b = nn.Sequential(
    nn.Conv2d(6, 16, 5),
    nn.ReLU(inplace=True),
    nn.AvgPool2d(kernel_size=2))
xf = torch.flatten(b(a(x)), 1)
print(xf.shape)
c = nn.Sequential(
    nn.Linear(16*4*4, 120),
    nn.ReLU(inplace=True),
    nn.Linear(120, 84),
    nn.ReLU(inplace=True))
d = nn.Linear(84, 10)

print(a(x).shape)
print(b(a(x)).shape)
print(c(xf).shape)
print(d(c(xf)).shape)
