import torch
import torchvision
import time
import random

from torch.utils.data import DataLoader, Dataset

root = './data/raw-data/'
data = torchvision.datasets.MNIST(
    root=root,
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False)

all_dataloader = DataLoader(
    dataset=data,
    batch_size=len(data),
    shuffle=False)

all_dataloader_ = list(all_dataloader)[0]


def data_loader(all_data, batch_size, shuffle):
    datas = all_data[0]
    targets = all_data[1]
    len_data = len(datas)
    idxs = [i for i in range(len_data)]
    if shuffle:
        random.shuffle(idxs)
    len_loader = len_data // batch_size
    if len_data - len_loader * batch_size:
        len_loader += 1
    for i in range(len_loader):
        idx = i * batch_size
        choised_idx = idxs[idx:idx + batch_size]
        yield [datas[choised_idx], targets[choised_idx]]


t = []
t.append(time.time())
dataloader_ = data_loader(all_data=all_dataloader_,
                          batch_size=100, shuffle=True)
for i, _ in enumerate(dataloader_):
    # print(i, data[0].shape, data[1].shape)
    t.append(time.time())

for i, t_ in enumerate(t):
    print(t_ - t[i-1])

all_dataloader1 = DataLoader(
    dataset=data,
    batch_size=100,
    shuffle=True)

t = []
t.append(time.time())

for _, _ in all_dataloader1:
    # print(i, data[0].shape, data[1].shape)
    t.append(time.time())

for i, t_ in enumerate(t):
    print(t_ - t[i-1])
