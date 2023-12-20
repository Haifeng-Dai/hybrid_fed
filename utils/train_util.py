import torch

from copy import deepcopy
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as f
from utils.lib_util import list_same_term


class DistillKL(torch.nn.Module):
    '''
    distilling loss
    '''

    def __init__(self, T):
        super(DistillKL, self).__init__()
        self.T = T

    def forward(self, logits_teacher, logits_student):
        prob_teacher = f.softmax(logits_teacher/self.T, dim=1)
        prob_student = f.log_softmax(logits_student/self.T, dim=1)
        loss = f.kl_div(prob_student, prob_teacher, reduction='batchmean')
        return loss * self.T**2 / logits_teacher.shape[0]


class EdgeServer:
    '''
    边缘服务器聚合
    '''

    def __init__(self, client_model):
        self.model = deepcopy(client_model[0])
        self.num_client = len(client_model)

        self.client_params = list_same_term(self.num_client)
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


def train_model(model, dataset, device='cpu', epochs=1):
    '''
    训练模型
    '''
    trained_model = deepcopy(model).to(device)
    trained_model.train()
    train_dataloader = DataLoader(
        dataset,
        batch_size=32,
        shuffle=True)
    optimizer = torch.optim.Adam(trained_model.parameters())
    loss_epoch = []
    for epoch in range(epochs):
        loss_epoch.append(0)
        for data, target in train_dataloader:
            optimizer.zero_grad()
            output = trained_model(data.to(device))
            loss = f.cross_entropy(output, target.to(device))
            loss.backward()
            optimizer.step()
            loss_epoch[epoch] += loss.item()
    return trained_model, loss_epoch


def train_model_disti(model, neighbor_server_model, weight, dataset, alpha, device='cpu', epochs=1, num_target=10):
    '''
    训练蒸馏模型
    '''
    trained_model = deepcopy(model).to(device)
    trained_model.train()
    train_dataloader = DataLoader(dataset, batch_size=320, shuffle=True)
    weight_device = weight.to(device)
    criterion = DistillKL(T=1)
    optimizer = torch.optim.Adam(trained_model.parameters())
    loss_epoch = []
    for epoch in range(epochs):
        loss_epoch.append(0)
        for data, target in train_dataloader:
            optimizer.zero_grad()
            logits = torch.zeros([len(target), num_target]).to(device)
            for j, server_model in enumerate(neighbor_server_model):
                logits += server_model(data.to(device)) * weight_device[j]
            logits.detach()
            output = trained_model(data.to(device))
            ce_loss = f.cross_entropy(output, target.to(device))
            distill_loss = criterion(output, logits)
            loss = alpha * ce_loss + (1 - alpha) * distill_loss
            loss.backward()
            optimizer.step()
            loss_epoch[epoch] += loss.item()
    return trained_model, loss_epoch


def eval_model(model, dataset, device):
    '''
    评估模型
    '''
    model_copy = deepcopy(model)
    model_copy.to(device)
    model_copy.eval()
    correct = 0
    data_loader = DataLoader(dataset, batch_size=320)
    for images, targets in data_loader:
        outputs = model_copy(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == targets.to(device)).sum()
    accuracy = correct / len(dataset)
    return accuracy.cpu()
