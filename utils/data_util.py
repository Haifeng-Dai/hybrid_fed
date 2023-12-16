import copy
import random
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


def get_dataset(dataset='mnist'):
    '''
    获取完整的原始数据集
    '''
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
    '''
    将数据集按标签分割
    input: dataset
    output: dict, key is target, value is the dataset of the key
    '''
    targets = set(dataset.targets.numpy().tolist())
    data_lib = dict.fromkeys(targets)
    for key in data_lib.keys():
        data_lib[key] = []
    for _, data in enumerate(dataset):
        data_lib[data[1]].append(data)
    return data_lib


def split_idx_evenly(idxs, num_set):
    '''
    把给定的指标集随机分割成几个集合
    input: index list
    out put: a list with indexes of each set
    '''
    idxs_copy = copy.deepcopy(idxs)
    random.shuffle(idxs_copy)
    idx_cut = idxs_copy[:len(idxs_copy)//num_set * num_set]
    idx_numpy = numpy.array(idx_cut)
    idx_set_matrix = idx_numpy.reshape(num_set, -1)
    idx_set = idx_set_matrix.tolist()
    return idx_set

def split_idx_proportion(idx, proportion):
    '''
    把给定指标集按给定比例分割
    input: two lists
    output: a list
    '''
    num_set = len(proportion)
    num_idx_set = list()
    for set in range(num_set):
        num_idx_set.append(round(len(idx) * proportion[set]))
    idx_copy = copy.deepcopy(idx)
    random.shuffle(idx_copy)
    idx_sets = []
    idx_left = idx_copy
    for set in range(num_set):
        idx_set = idx_left[:num_idx_set[set]]
        idx_left = idx_left[num_idx_set[set]:]
        idx_sets.append(idx_set)
    return idx_sets


def idx_split(dataset, mode='iid', n_dataset=1, n_data_each_set=1):
    '''
    分割数据集
    '''
    targets_list = dataset.targets.tolist()
    all_targets = set(targets_list)
    idx_target = dict()
    for target in all_targets:
        idx_target[target] = list()
        for idx, target_in_list in enumerate(targets_list):
            if target_in_list == target:
                idx_target[target] += [idx]
    # 独立同分布
    if mode == 'iid':
        if n_dataset * n_data_each_set > len(dataset):
            raise ValueError(
                f'number of client ({n_dataset}) times number of data of each client ({n_data_each_set}) no more than number of total data ({len(dataset)})')
        n_each_set = dict()
        for target in all_targets:
            n_each_set[target] = int(
                len(idx_target[target]) / len(targets_list) * n_data_each_set)
        dataset_splited = dict()
        left_idx_target = idx_target
        for i in range(n_dataset):
            dataset_splited[i] = list()
            for target in all_targets:
                choiced_idx = numpy.random.choice(
                    left_idx_target[target],
                    n_each_set[target],
                    replace=False)
                dataset_splited[i] += list(choiced_idx)
                left_idx_target[target] = list(
                    set(left_idx_target[target]) - set(dataset_splited[i]))
        return dataset_splited
    elif mode == 'partial-iid':
        print('TO DO.')
