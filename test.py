import torch
import torchvision
import time

from torch.utils.data import DataLoader, Dataset

t = []
t.append(time.time())
root = './data/raw-data/'
data = torchvision.datasets.MNIST(
    root=root,
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False)

dataloader = DataLoader(
    dataset=data,
    batch_size=100,
    shuffle=False)

for i in dataloader:
    t.append(time.time())

for i, t_ in enumerate(t[1:]):
    print(t_ - t[i-1])

