import torch
import numpy
import copy

from torch.utils.data import Dataset, DataLoader


class Loss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return torch.nn.functional.cross_entropy(input, target)


class Server:
    def __init__(self, model, client_params):
        self.model = copy.deepcopy(model)
        self.client_params = client_params
        self.num_client = len(self.client_params)
        self.parameters = self.client_params[0]

        self.fed_avg()
        self.model.load_state_dict(self.parameters)

    def fed_avg(self):
        for client in range(1, self.num_client):
            for key in self.parameters:
                new_params = self.client_params[client][key]
                # print(new_params.equal(self.server_params[key]), end=' | ')
                self.parameters[key] = self.parameters[key].add(
                    new_params)
                # tmp_1 = copy.deepcopy(new_params)
                # tmp_2 = copy.deepcopy(self.parameters[key])
                # print(new_params.equal(tmp_2.div(2)))
        for key in self.parameters:
            self.parameters[key] = self.parameters[key].div(2)


class DealDataset(Dataset):
    def __init__(self, dataset, idx):
        self.dataset = dataset
        self.idx = idx
        self.len = len(idx)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        img, target = self.dataset[self.idx[index]]
        return img, target


def idx_split(dataset, mode='iid', n_dataset=1, n_data_each_set=1):
    labels_list = dataset.targets.tolist()
    all_labels = set(labels_list)
    idx_label = dict()
    for label in all_labels:
        idx_label[label] = list()
        for idx, label_in_list in enumerate(labels_list):
            if label_in_list == label:
                idx_label[label] += [idx]

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


def train_model(model, dataset, device='cpu', epochs=1, tqdm_position=0):
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

        #     if (i+1) % 100 == 0:
        #         print('\r', end='')
        #         print(
        #             f'step [{i+1}/{len(train_dataloader)}], loss: {loss.item():.4f}', end='')
        # print(f'\nepoch {epoch+1}/{epochs} down.')
    return trained_model


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