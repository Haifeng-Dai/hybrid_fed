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


class EdgeServer:
    '''
    边缘服务器聚合
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


def train_model(model, dataset, device='cpu', epochs=1):
    '''
    训练模型
    '''
    trained_model = copy.deepcopy(model).to(device)
    trained_model.train()
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(trained_model.parameters())
    for epoch in range(epochs):
        for i, (data, target) in enumerate(train_dataloader):
            optimizer.zero_grad()
            output = trained_model(data.to(device))
            loss = criterion(output, target.to(device))
            loss.backward()
            optimizer.step()
    return trained_model


def eval_model(model, dataset, device):
    '''
    评估模型
    '''
    server_model = copy.deepcopy(model)
    server_model.eval()
    server_model.to(device)
    with torch.no_grad():
        correct = 0
        total = 0
        data_loader = DataLoader(dataset, batch_size=32, shuffle=False)
        for images, targets in data_loader:
            outputs = server_model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets.to(device)).sum().item()
        print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
