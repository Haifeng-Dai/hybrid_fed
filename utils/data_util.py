import torch
import torchvision
import random
import numpy

from utils.lib_util import *
from torch.utils.data import DataLoader


# def data_loader(dataset, batch_size, shuffle, ):
#     '''
#     根据给定的指标集返回一个数据集
#     '''

#     return DataLoader(
#         dataset=dataset,
#         batch_size=batch_size,
#         shuffle=True)


def get_dataset(dataset='mnist'):
    '''
    获取完整的原始数据集
    '''
    raw_data = './data/'
    if dataset == 'mnist':
        train_dataset = torchvision.datasets.MNIST(
            root=raw_data,
            transform=torchvision.transforms.ToTensor(),
            train=True,
            download=True)
        test_dataset = torchvision.datasets.MNIST(
            root=raw_data,
            transform=torchvision.transforms.ToTensor(),
            train=False)
    elif dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root=raw_data,
            transform=torchvision.transforms.ToTensor(),
            train=True,
            download=True)
        test_dataset = torchvision.datasets.CIFAR10(
            root=raw_data,
            transform=torchvision.transforms.ToTensor(),
            train=False)
    elif dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(
            root=raw_data,
            transform=torchvision.transforms.ToTensor(),
            train=True,
            download=True)
        test_dataset = torchvision.datasets.CIFAR100(
            root=raw_data,
            transform=torchvision.transforms.ToTensor(),
            train=False)
    else:
        raise ValueError('dataset error.')
    [c, h, w] = train_dataset[0][0].shape
    return train_dataset, test_dataset, c, h, w


def split_dataset(train_dataset_o, target_list, args):
    TrainDatasetSplited = SplitData(train_dataset_o)
    all_target = TrainDatasetSplited.targets
    num_target = TrainDatasetSplited.num_target

    # client_main_target = numpy.random.choice(
    #     all_target, args.num_all_client, replace=False).tolist()
    # train_dataset_client = TrainDatasetSplited.server_non_iid(
    #     num_server=args.num_all_server,
    #     num_server_client=num_server_client,
    #     num_client_data=args.num_client_data,
    #     client_main_target=client_main_target,
    #     proportion=args.proportion)
    train_dataset_client = TrainDatasetSplited.server_target_niid(
        num_server=args.num_all_server,
        num_server_client=args.num_all_client // args.num_all_server,
        num_client_data=args.num_client_data,
        target_list=target_list)
    train_dataloader = list_same_term(args.num_all_client)
    validate_dataloader = list_same_term(args.num_all_client)
    for i, dataset_ in enumerate(train_dataset_client):
        # print()
        [dataset_train, dataset_test] = split_parts_random(
            dataset_, [1000, 200])
        train_dataloader[i] = DataLoader(
            dataset=dataset_train,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=args.num_workers)
        validate_dataloader[i] = DataLoader(
            dataset=dataset_test,
            batch_size=args.batch_size,
            shuffle=True,
            pin_memory=True,
            num_workers=args.num_workers)
    return num_target, train_dataloader, validate_dataloader


class SplitData:
    '''
    分割数据集
    '''

    def __init__(self, dataset):
        self.initial_dataset = dataset
        self.targets = self.get_target()
        self.num_target = len(self.targets)

    def get_target(self):
        # 获取所有标签
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
        # 将数据集按标签分割
        targets = self.targets
        splited_data = dict.fromkeys(targets)
        for key in splited_data.keys():
            splited_data[key] = []
        for data in self.initial_dataset:
            splited_data[data[1]].append(data)
        for key in splited_data.keys():
            random.shuffle(splited_data[key])
        return splited_data

    def num_data_target(self):
        # 获取每个标签对数据集的数量
        num_data_target_all = []
        splited_data = self.split_data()
        for target in self.targets:
            num_data_target_all.append(len(splited_data[target]))
        return num_data_target_all

    def all_iid(self, num_client, num_client_data):
        # 按照客户端数量和每个客户端的数据量分配数据
        # if num_client_data * num_client > self.num_target * min(self.num_data_target()):
        #     raise ValueError('too large num_client_data * num_client.')
        # if num_client_data % self.num_target != 0:
        #     raise ValueError('num_client_data \% num_targets != 0.')

        num_data_target = num_client_data // self.num_target
        client_data = list_same_term(num_client)
        splited_data = self.split_data()
        for target in self.targets:
            data_target = deepcopy(splited_data[target])
            random.shuffle(data_target)
            idx = 0
            for client in range(num_client):
                add_data = data_target[idx: idx + num_data_target]
                client_data[client].extend(add_data)
                idx += num_data_target
        return client_data

    def all_non_iid(self, num_client, num_client_data, client_main_target, proportion=None):
        # if num_client_data * num_client > self.num_target * min(self.num_data_target()):
        #     raise ValueError('too large num_client_data * num_client.')
        if not proportion:
            proportion = 2 / self.num_target
        # if num_client_data * num_client * proportion > min(self.num_data_target()):
        #     raise Warning(
        #         'maybe too large num_client_data * num_client * proportion.')
        # if num_client_data % self.num_target != 0:
        #     raise ValueError('num_client_data \% num_targets != 0.')

        num_client_data_minor = int(
            (1 - proportion) * num_client_data // (self.num_target - 1))
        num_client_data_mian = num_client_data - \
            num_client_data_minor * (self.num_target - 1)
        splited_data = self.split_data()
        client_data = list_same_term(num_client)
        for target in self.targets:
            data_target = deepcopy(splited_data[target])
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

    def server_non_iid(self, num_server, num_server_client, num_client_data, client_main_target, proportion=None):
        # 按照客户端数量和每个客户端的数据量分配数据
        # if num_client_data * num_server * num_server_client > self.num_target * min(self.num_data_target()):
        #     raise ValueError(
        #         'too large num_client_data * num_server * num_server_client.')
        if not proportion:
            proportion = 2 / self.num_target
        num_data_server = num_server_client * num_client_data
        # if num_server_client * num_client_data * num_server * proportion > min(self.num_data_target()):
        #     raise Warning(
        #         'maybe too large num_server * num_server_client * num_client_data.')
        # if num_client_data % self.num_target != 0:
        #     raise ValueError('num_client_data \% num_targets != 0.')

        server_data = self.all_non_iid(
            num_server, num_data_server, client_main_target, proportion)
        server_client_data = list_same_term(num_server)
        for server in range(num_server):
            server_data_ = server_data[server]
            random.shuffle(server_data_)
            idx = 0
            server_client_data[server] = list_same_term(num_server_client)
            for client in range(num_server_client):
                add_data = server_data_[idx: idx + num_client_data]
                server_client_data[server][client].extend(add_data)
                idx += num_client_data
        client_data = []
        for server in range(num_server):
            for client in range(num_server_client):
                client_data.append(server_client_data[server][client])
        return client_data

    def client_non_iid(self, num_server, num_server_client, num_client_data, client_main_target, proportion=None):
        # if num_client_data * num_server * num_server_client > self.num_target * min(self.num_data_target()):
        #     raise ValueError(
        #         'too large num_client_data * num_server * num_server_client.')
        if not proportion:
            proportion = 2 / self.num_target
        # if num_server_client * num_client_data * num_server * proportion > min(self.num_data_target()):
        #     raise Warning(
        #         'maybe too large num_server * num_server_client * num_client_data.')
        # if num_client_data % self.num_target != 0:
        #     raise ValueError('num_client_data \% num_targets != 0.')

        num_client_data_minor = int(
            (1 - proportion) * num_client_data // (self.num_target - 1))
        num_client_data_mian = num_client_data - \
            num_client_data_minor * (self.num_target - 1)
        splited_data = deepcopy(self.split_data())
        server_client_data = list_same_term(
            num_server, list_same_term(num_server_client))
        for target in range(self.num_target):
            data_target = splited_data[target]
            idx = 0
            for server in range(num_server):
                for client in range(num_server_client):
                    if client_main_target[client] == target:
                        add_data = data_target[idx: idx + num_client_data_mian]
                        server_client_data[server][client].extend(add_data)
                        idx += num_client_data_mian
                        continue
                    add_data = data_target[idx: idx + num_client_data_minor]
                    server_client_data[server][client].extend(add_data)
                    idx += num_client_data_minor
        client_data = []
        for server in range(num_server):
            for client in range(num_server_client):
                client_data.append(server_client_data[server][client])
        return client_data

    def server_target_niid(self, num_server, num_server_client, target_list, num_client_data):
        server_data = list_same_term(num_server)
        splited_data = deepcopy(self.split_data())
        num_server_data = num_client_data * num_server_client
        for server in range(num_server):
            num_data_each_target = num_server_data // len(target_list[server])
            # print(num_data_each_target)
            for target in target_list[server]:
                add_data = splited_data[target][
                    : num_data_each_target]
                server_data[server].extend(add_data)
                splited_data[target] = splited_data[target][
                    num_data_each_target:]
        client_data = list_same_term(num_server * num_server_client)
        client_idx = 0
        for server in range(num_server):
            random.shuffle(server_data[server])
            for _ in range(num_server_client):
                client_data[client_idx] = server_data[server][: num_client_data]
                server_data[server] = server_data[server][num_client_data:]
                client_idx += 1
        return client_data


def split_parts_random(dataset, num_list):
    '''
    split dataset with the list of number of each sub-dataset
    '''
    client_data = []
    dataset_copy = deepcopy([dataset[i] for i in range(len(dataset))])
    random.shuffle(dataset_copy)
    idx = 0
    for num_data in num_list:
        client_data.append(dataset_copy[idx: idx + num_data])
        idx += num_data
    return client_data


# if __name__ == '__main__':

#     dataset, _ = get_dataset()
#     DataSplit = SplitData(dataset=dataset)
#     splited_data = DataSplit.split_data()
#     client_data = DataSplit.client_non_iid(2, 3, 100)
#     print(type(client_data), len(client_data))
#     print(type(client_data[0]), len(client_data[0]))
