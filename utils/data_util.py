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
            download=True)
        test_dataset = torchvision.datasets.MNIST(
            root=raw_data,
            train=False,
            transform=torchvision.transforms.ToTensor())
    elif dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root=raw_data,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True)
        test_dataset = torchvision.datasets.CIFAR10(
            root=raw_data,
            train=False,
            transform=torchvision.transforms.ToTensor())
    elif dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(
            root=raw_data,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True)
        test_dataset = torchvision.datasets.CIFAR100(
            root=raw_data,
            train=False,
            transform=torchvision.transforms.ToTensor())
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


def train_data_split(dataset_splited, all_client):
    num_all_client = len(all_client)
    all_target = dataset_splited.keys()
    idx_client_target = []
    for client in all_client:
        idx_client_target.append([])
    for target in all_target:
        num_idx_dataset_target = len(dataset_splited[target])
        idxs_dataset_target = [i for i in range(num_idx_dataset_target)]
        idx_target_client = split_idx_evenly(
            idxs_dataset_target, num_all_client)
        for client in all_client:
            idx_client_target[client].append(idx_target_client[client])
    return idx_client_target


def idx_to_dataset(dataset, idxs):
    '''
    返回对应指标集的数据子集
    '''
    return [dataset[idx] for idx in idxs]
