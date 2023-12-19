import torch

from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
from utils.lib_util import get_list


class LossWithoutDistillation(torch.nn.Module):
    '''
    Loss Without Distillation
    '''

    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        ce_loss = torch.nn.functional.cross_entropy(reduction='sum')
        total_loss = ce_loss(input, target)
        return total_loss


class LossWithDistillation(torch.nn.Module):
    '''
    Loss With Distillation
    '''

    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, input, target, logits):
        ce_loss = torch.nn.functional.cross_entropy(input, target, reduction='sum')
        kl_loss = torch.nn.functional.kl_div(input, logits, reduction='batchmean')
        total_loss = self.alpha * ce_loss + self.beta * kl_loss
        return total_loss


class EdgeServer:
    '''
    边缘服务器聚合
    '''

    def __init__(self, client_model):
        self.model = deepcopy(client_model[0])
        self.num_client = len(client_model)

        self.client_params = get_list(self.num_client)
        # for client in range(self.num_client):
        #     self.client_params.append([])
        i = 0
        for client in client_model:
            self.client_params[i] = client.state_dict()
            i += 1

    # 平均
    def average(self):
        model = deepcopy(self.model)
        parameters = deepcopy(self.client_params[0])
        # print(next(parameters).device)
        for client in range(1, self.num_client):
            for key in parameters:
                parameters[key] += self.client_params[client][key]
        for key in parameters:
            parameters[key] /= self.num_client
        model.load_state_dict(parameters)
        return model

    # 加权平均
    def weighted_average(self, weight):
        model = deepcopy(self.model)
        parameters = deepcopy(self.client_params[0])
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
    trained_model = deepcopy(model).to(device)
    trained_model.train()
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    criterion = LossWithoutDistillation()
    optimizer = torch.optim.Adam(trained_model.parameters())
    loss_sum = 0
    for epoch in range(epochs):
        for data, target in train_dataloader:
            optimizer.zero_grad()
            output = trained_model(data.to(device))
            loss = criterion(output, target.to(device))
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
    return trained_model, loss_sum


def train_model_disti(model, neighbor_server_model, weight, dataset, alpha, beta, device='cpu', epochs=1, num_target=10):
    '''
    训练蒸馏模型
    '''
    batch_size = 32
    trained_model = deepcopy(model).to(device)
    trained_model.train()
    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    criterion = LossWithDistillation(alpha, beta)
    optimizer = torch.optim.Adam(trained_model.parameters())
    loss_sum = 0
    for epoch in range(epochs):
        for data, target in train_dataloader:
            optimizer.zero_grad()
            logits = torch.zeros([len(target), num_target]).to(device)
            for j, server_model in enumerate(neighbor_server_model):
                logits += server_model(data.to(device)) * weight[j]
            logits.detach()
            output = trained_model(data.to(device))
            loss = criterion(output, target.to(device), logits)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item()
    return trained_model, loss_sum


def eval_model(model, dataset, device):
    '''
    评估模型
    '''
    model_copy = deepcopy(model).eval().to(device)
    correct = 0
    data_loader = DataLoader(dataset, batch_size=32)
    for images, targets in data_loader:
        outputs = model_copy(images.to(device))
        _, predicted = torch.max(outputs, 1)
        correct += torch.eq(predicted, targets.to(device)).sum()
    # print(f'Test Accuracy: {100 * correct / total:.2f}%')
    accuracy = correct / len(dataset)
    return accuracy
