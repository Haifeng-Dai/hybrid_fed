import torch
import numpy
import copy
import torchvision

from torch.utils.data import Dataset, DataLoader


class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return torch.nn.functional.cross_entropy(input, target)


class Server:
    '''
    服务器聚合策略
    '''

    def __init__(self, model, client_params):
        self.model = copy.deepcopy(model)
        self.client_params = copy.deepcopy(client_params)
        self.num_client = len(self.client_params)

    # 平均
    def average(self):
        model = copy.deepcopy(self.model)
        parameters = copy.deepcopy(self.client_params[0])
        for client in range(1, self.num_client):
            for key in parameters:
                parameters[key] += self.client_params[client][key]
        for key in parameters:
            parameters[key] /= self.num_client
        model.load_state_dict(parameters)
        return model

    # 加权平均
    def weighted_average(self, weight):
        model = copy.deepcopy(self.model)
        parameters = copy.deepcopy(self.client_params[0])
        for key in parameters:
            parameters[key] *= weight[0]
        for client in range(1, self.num_client):
            for key in parameters:
                parameters[key] += self.client_params[client][key] * \
                    weight[client]
        model.load_state_dict(parameters)
        return model


class DealDataset(Dataset):
    '''
    根据给定的指标集返回一个Dataset类，即数据集
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


# 获取数据集
def get_dataset(dataset='mnist'):
    if dataset == 'mnist':
        train_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True
        )
        test_dataset = torchvision.datasets.MNIST(
            root='./data',
            train=False,
            transform=torchvision.transforms.ToTensor()
        )
    elif dataset == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='./data',
            train=False,
            transform=torchvision.transforms.ToTensor()
        )
    elif dataset == 'cifar100':
        train_dataset = torchvision.datasets.CIFAR100(
            root='./data',
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True
        )
        test_dataset = torchvision.datasets.CIFAR100(
            root='./data',
            train=False,
            transform=torchvision.transforms.ToTensor()
        )
    else:
        raise ValueError('dataset error.')
    return train_dataset, test_dataset


# 根据标签分割数据集
def idx_split(dataset, mode='iid', n_dataset=1, n_data_each_set=1):
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

# 训练模型
def train_model(model, dataset, device='cpu', epochs=1):
    trained_model = copy.deepcopy(model).to(device)
    trained_model.train()
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(trained_model.parameters())
    for epoch in range(epochs):
        for i, (data, label) in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = trained_model(data.to(device))
            loss = criterion(output, label.to(device))
            loss.backward()
            optimizer.step()
    return trained_model

# 评估模型
def eval_model(model, dataset, device):
    server_model = copy.deepcopy(model)
    server_model.eval()
    server_model.to(device)
    with torch.no_grad():
        correct = 0
        total = 0
        data_loader = DataLoader(dataset, batch_size=32, shuffle=True)
        for images, labels in data_loader:
            outputs = server_model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()
        print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
