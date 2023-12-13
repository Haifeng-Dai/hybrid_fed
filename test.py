import torch
import torchvision
import numpy

from torch.utils.data import Dataset, DataLoader

torch.set_printoptions(
    precision=1,  # 精度，保留小数点后几位，默认4
    threshold=100000,  # 最大数据量
    edgeitems=3,  # 在缩略显示时在起始和默认显示的元素个数
    linewidth=150,  # 每行最多显示的字符数，默认80，超过则换行显示
    profile=None,  # 3种预定义的显示模板，可选’default’、‘short’、‘full’
    sci_mode=False  # 用科学技术法显示数据，默认True
)


train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

data = train_dataset.data[0]
print(data, type(data))
print(data.shape)