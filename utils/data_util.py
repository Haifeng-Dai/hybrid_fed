import torchvision

from copy import deepcopy
from torch.utils.data import Dataset
from lib_util import split_idx_evenly, split_idx_proportion


class DealDataset(Dataset):
    '''
    根据给定的指标集返回一个数据集
    '''

    def __init__(self, dataset, idx):
        self.__dataset__ = dataset
        self.__idx__ = idx

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, index):
        return self.__dataset__[self.__idx__[index]]


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


def train_data_split(dataset_splited, all_client, mode='iid', proportion=None):
    '''
    把数据集分配给所以客户端
    '''
    if mode == 'iid':
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
    if mode == 'proportion':
        if proportion == None:
            raise ValueError('proportion is required.')
        num_all_client = len(all_client)
        all_target = dataset_splited.keys()
        idx_client_target = []
        for client in all_client:
            idx_client_target.append([])
        for target in all_target:
            num_idx_dataset_target = len(dataset_splited[target])
            idxs_dataset_target = [i for i in range(num_idx_dataset_target)]
            idx_target_client = split_idx_proportion(
                idxs_dataset_target, proportion)
            for client in all_client:
                idx_client_target[client].append(idx_target_client[client])
        return idx_client_target
