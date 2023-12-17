import torch

from copy import deepcopy
from torch.utils.data import Dataset, DataLoader


class LossWithoutDistillation(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        ce_loss = torch.nn.CrossEntropyLoss()
        total_loss = ce_loss(input, target)
        return total_loss


class LossWithDistillation(torch.nn.Module):
    def __init__(self, alpha, beta):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, input, target, logit):
        ce_loss = torch.nn.CrossEntropyLoss()
        kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
        total_loss = self.alpha * \
            ce_loss(input, target) + self.beta * kl_loss(input, logit)
        return total_loss


class EdgeServer:
    '''
    边缘服务器聚合
    '''

    def __init__(self, client_model):
        self.model = deepcopy(client_model[0])
        self.num_client = len(client_model)

        self.client_params = []
        for client in range(self.num_client):
            self.client_params.append([])
        i = 0
        for client in client_model:
            self.client_params[i] = client.state_dict()
            i += 1

    # 平均
    def average(self):
        model = deepcopy(self.model)
        parameters = deepcopy(self.client_params[0])
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


def train_model(model, dataset, criterion, device='cpu', epochs=1):
    '''
    训练模型
    '''
    trained_model = deepcopy(model).to(device)
    trained_model.train()
    train_dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    # criterion = torch.nn.CrossEntropyLoss()
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
    model_copy = deepcopy(model)
    model_copy.eval()
    model_copy.to(device)
    with torch.no_grad():
        correct = 0
        total = 0
        data_loader = DataLoader(dataset, batch_size=32)
        for images, targets in data_loader:
            outputs = model_copy(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += (predicted == targets.to(device)).sum().item()
        print('Test Accuracy: {:.2f}%'.format(100 * correct / total))
