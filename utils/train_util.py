import torch

from copy import deepcopy
import torch.nn.functional as f

from utils.lib_util import list_same_term


class DistillKL(torch.nn.Module):
    '''
    distilling loss
    '''

    def __init__(self, T, alpha):
        super(DistillKL, self).__init__()
        self.T = T
        self.alpha = alpha

    def forward(self, output, target, logits_teacher):
        prob_teacher = f.softmax(logits_teacher/self.T, dim=1)
        prob_student = f.softmax(output/self.T, dim=1)
        soft_loss = f.kl_div(prob_student, prob_teacher, reduction='batchmean')
        hard_loss = f.cross_entropy(output, target)
        loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss * self.T**2 / logits_teacher.shape[0]
        return loss


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


def train_model(model, dataloader, device='cpu'):
    '''
    训练模型
    '''
    trained_model = deepcopy(model).to(device)
    trained_model.train()
    optimizer = torch.optim.Adam(trained_model.parameters())
    for data, target in dataloader:
        optimizer.zero_grad()
        output = trained_model(data.to(device))
        loss = f.cross_entropy(output, target.to(device))
        loss.backward()
        optimizer.step()
    return trained_model

def train_model_disti_weighted(model, neighbor_server_model, weight, dataloader, alpha, T, device='cpu', num_target=10):
    '''
    训练蒸馏模型, logits加权聚合
    '''
    trained_model = deepcopy(model).to(device)
    trained_model.train()
    weight_device = weight.to(device)
    optimizer = torch.optim.Adam(trained_model.parameters())
    criterion = DistillKL(T=T, alpha=alpha)
    for data, target in dataloader:
        data_device = data.to(device)
        teacher_logits = torch.zeros([len(target), num_target], device=device)
        for i, model in enumerate(neighbor_server_model):
            teacher_model = deepcopy(model).to(device)
            teacher_model.eval()
            teacher_logits += teacher_model(data_device) * weight_device[i]
        optimizer.zero_grad()
        output = trained_model(data_device)
        loss = criterion(output, target.to(device), teacher_logits)
        loss.backward()
        optimizer.step()
    return trained_model

def train_model_disti_single(model, teacher_model, dataloader, alpha, T, device='cpu'):
    '''
    训练蒸馏模型, 单个teacher
    '''
    trained_model = deepcopy(model).to(device)
    trained_model.train()
    teacher_model = deepcopy(teacher_model).to(device)
    teacher_model.eval()
    criterion = DistillKL(T=T, alpha=alpha)
    optimizer = torch.optim.Adam(trained_model.parameters())
    for data, target in dataloader:
        optimizer.zero_grad()
        logits = teacher_model(data.to(device))
        output = trained_model(data.to(device))
        loss = criterion(output, target.to(device), logits)
        loss.backward()
        optimizer.step()
    return trained_model


def eval_model(model, dataloader, device):
    '''
    评估模型
    '''
    model_copy = deepcopy(model)
    model_copy.to(device)
    model_copy.eval()
    correct = 0
    len_data = 0
    for images, targets in dataloader:
        outputs = model_copy(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == targets.to(device)).sum()
        len_data += len(targets)
    accuracy = correct / len_data
    return accuracy.cpu()


# if __name__ == '__main__':
#     loss_fun = DistillKL(T=1)
#     a = torch.randn(1, 5)
#     b = torch.randn(1, 5)
#     print(a)
#     print(b)
#     loss_fun_1 = torch.nn.CrossEntropyLoss()
#     c = loss_fun_1(a, b)
#     print(c)

#     d = loss_fun(a, b)
#     print(d)
