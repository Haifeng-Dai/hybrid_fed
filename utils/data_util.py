import torch
import torchvision
import numpy

from torch.utils.data import Dataset
from utils.lib_util import *


class DealDataset(Dataset):
    '''
    根据给定的指标集返回一个数据集
    '''

    def __init__(self, dataset, idx):
        self.__dataset__ = dataset
        self.__idx__ = idx

    def __len__(self):
        return len(self.__idx__)

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


class SplitData:
    '''
    分割数据集
    '''

    def __init__(self, dataset):
        self.initial_dataset = dataset
        self.targets = self.get_target()
        self.num_target = len(self.targets)
        # self.num_data = len(self.initial_dataset)

    # 将数据集按标签分割
    def get_target(self):
        if isinstance(self.initial_dataset.targets, list):
            targets = set(self.initial_dataset.targets)
        elif torch.is_tensor(self.initial_dataset.targets):
            targets = set(self.initial_dataset.targets.numpy().tolist())
        else:
            raise ValueError('dataset.targets is not tensor or list.')
        targets = list(targets)
        targets.sort()
        return targets

    def split_data(self):
        targets = self.targets
        splited_data = dict.fromkeys(targets)
        for key in splited_data.keys():
            splited_data[key] = []
        for data in self.initial_dataset:
            splited_data[data[1]].append(data)
        return splited_data

    # 获取每个标签对数据集的数量
    def num_data_target(self):
        num_data_target_all = []
        splited_data = self.split_data()
        for target in self.targets:
            num_data_target_all.append(len(splited_data[target]))
        return num_data_target_all

    # 按照客户端数量和每个客户端的数据量分配数据
    def all_iid(self, num_client, num_client_data):
        if num_client_data > min(self.num_data_target()):
            raise ValueError('too large num_target.')
        if num_client_data % self.num_target != 0:
            raise ValueError('num_client_data / num_targets != 0.')
        num_data_target = num_client_data // self.num_target
        client_data = list_same_term(num_client)
        splited_data = self.split_data()
        for target in self.targets:
            data_target = splited_data[target]
            idx = 0
            random.shuffle(data_target)
            for client in range(num_client):
                add_data = data_target[idx: idx + num_data_target]
                client_data[client].extend(add_data)
                idx += num_data_target
        return client_data

    def all_non_iid(self, num_client, num_client_data, proportion=None):
        if proportion:
            proportion = 2 / self.num_target
        if num_client <= self.num_target:
            replace = False
        else:
            replace = True
        num_client_data_minor = (1 - proportion) * \
            num_client_data // (self.num_target - 1)
        num_client_data_mian = num_client_data - num_client_data_minor
        client_main_target = numpy.random.choice(
            self.targets, num_client, replace=replace).tolist()
        splited_data = self.split_data()
        client_data = list_same_term(num_client)
        for target in self.targets:
            data_target = splited_data[target]
            random.shuffle(data_target)
            idx = 0
            for client in range(num_client):
                if client_main_target[client] == target:
                    add_data = data_target[idx: idx + num_client_data_mian]
                    client_data[client].extend(add_data)
                    idx += num_client_data_mian
                    continue
                add_data = data_target[idx: idx + num_client_data_minor]
                client_data[client].extend(add_data)
                idx += num_client_data_minor
        return client_data

    # 按照客户端数量和每个客户端的数据量分配数据
    def server_non_iid(self, num_server, num_server_client, num_client_data, proportion=None):
        if proportion:
            proportion = 2 / self.num_target
        num_data_server = num_server_client * num_client_data
        server_data = self.all_non_iid(
            num_server, num_data_server, proportion)
        server_client_data = list_same_term(num_server)
        for server in range(num_server):
            server_data_ = server_data(server)
            random.shuffle(server_data_)
            idx = 0
            server_client_data[server] = list_same_term(num_server_client)
            for client in range(num_server_client):
                add_data = server_data_[idx: idx + num_client_data]
                server_client_data[server][client].extend(add_data)
                idx += num_client_data
        client_date = list_same_term(num_server * num_server_client)
        for server in range(num_server):
            for client in range(num_server_client):
                client_date.append(server_client_data[server][client])
        return client_date

    def client_non_iid(self, num_server, num_server_client, num_client_data, proportion=None):
        if proportion:
            proportion = 2 / self.num_target
        num_client_data_minor = (1 - proportion) * \
            num_client_data // (self.num_target - 1)
        num_client_data_mian = num_client_data - num_client_data_minor
        if num_server_client <= self.num_target:
            replace = False
        else:
            replace = True
        client_main_target = numpy.random.choice(
            self.targets, num_server_client, replace=replace).tolist()
        server_client_main_target = list_same_term(
            num_server, client_main_target)
        server_client_data = list_same_term(num_server)
        splited_data = self.split_data()
        for target in range(self.num_target):
            data_target = splited_data[target]
            idx = 0
            for server in range(num_server):
                server_client_data[server] = list_same_term(num_server_client)
                for client in range(num_server_client):
                    if server_client_main_target[server][client] == target:
                        server_client_data[server][client].append(
                            data_target[idx: idx + num_client_data_mian])
                        idx += num_client_data_mian
                        continue
                    server_client_data[server][client].append(
                        data_target[idx: idx + num_client_data_minor])
                    idx += num_client_data_mian
        client_date = list_same_term(num_server * num_server_client)
        for server in range(num_server):
            for client in range(num_server_client):
                client_date.append(server_client_data[server][client])
        return client_date
