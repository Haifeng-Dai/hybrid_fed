import torch
import numpy
import torchvision

torch.set_printoptions(
    precision=1,  # 精度，保留小数点后几位，默认4
    threshold=100000,  # 最大数据量
    edgeitems=3,  # 在缩略显示时在起始和默认显示的元素个数
    linewidth=150,  # 每行最多显示的字符数，默认80，超过则换行显示
    profile=None,  # 3种预定义的显示模板，可选’default’、‘short’、‘full’
    sci_mode=False  # 用科学技术法显示数据，默认True
)


class DealDataset(torch.utils.data.Dataset):
    '''
    根据给定的指标集返回一个数据集
    '''

    def __init__(self, dataset, idx):
        self.dataset = dataset
        self.idx = idx
        self.len = len(idx)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img, target = self.dataset[self.idx[index]]
        return img, target


raw_data = './data/raw-data'
dataset = torchvision.datasets.MNIST(
    root=raw_data,
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

labels = set(dataset.train_labels.numpy().tolist())
data_lib = dict.fromkeys(labels, list())

for i in range(len(labels)):
    data_lib[i] = [i,]
print(data_lib)

for i, data in enumerate(dataset):
    if i < 10:
        data_lib[data[1]].append(data)
    else:
        break

print(data_lib)
# pprint.pprint(len(data_lib[0]))