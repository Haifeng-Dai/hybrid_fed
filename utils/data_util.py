import torchvision
import numpy

from torch.utils.data import Dataset


class DealDataset(Dataset):
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


# 获取完整的原始数据集
def get_dataset(dataset='mnist'):
    raw_data = './data/raw-data'
    if dataset == 'mnist':
        train_dataset = torchvision.datasets.MNIST(
            root=raw_data,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True
        )
        test_dataset = torchvision.datasets.MNIST(
            root=raw_data,
            train=False,
            transform=torchvision.transforms.ToTensor()
        )
    elif dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root=raw_data,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root=raw_data,
            train=False,
            transform=torchvision.transforms.ToTensor()
        )
    elif dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(
            root=raw_data,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root=raw_data,
            train=False,
            transform=torchvision.transforms.ToTensor()
        )
    else:
        raise ValueError('dataset error.')
    return train_dataset, test_dataset

def split_data(dataset):
    labels = set(dataset.train_labels.numpy().tolist())
    data_lib = dict.fromkeys(labels, list())
    for _, data in enumerate(dataset):
        data_lib[data[1]].append(data)



def idx_split(dataset, mode='iid', n_dataset=1, n_data_each_set=1):
    '''
    根据标签分割数据集
    '''
    labels_list = dataset.targets.tolist()
    all_labels = set(labels_list)
    idx_label = dict()
    for label in all_labels:
        idx_label[label] = list()
        for idx, label_in_list in enumerate(labels_list):
            if label_in_list == label:
                idx_label[label] += [idx]
    # 独立同分布
    if mode == 'iid':
        if n_dataset * n_data_each_set > len(dataset):
            raise ValueError(
                f'number of client ({n_dataset}) times number of data of each client ({n_data_each_set}) no more than number of total data ({len(dataset)})')
        n_each_set = dict()
        for label in all_labels:
            n_each_set[label] = int(
                len(idx_label[label]) / len(labels_list) * n_data_each_set)
        dataset_splited = dict()
        left_idx_label = idx_label
        for i in range(n_dataset):
            dataset_splited[i] = list()
            for label in all_labels:
                choiced_idx = numpy.random.choice(
                    left_idx_label[label],
                    n_each_set[label],
                    replace=False)
                dataset_splited[i] += list(choiced_idx)
                left_idx_label[label] = list(
                    set(left_idx_label[label]) - set(dataset_splited[i]))
        return dataset_splited
    elif mode == 'partial-iid':
        print('TO DO.')
