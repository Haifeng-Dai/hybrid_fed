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
        loss = self.alpha * hard_loss + \
            (1 - self.alpha) * soft_loss * self.T**2 / logits_teacher.shape[0]
        return loss

def server_communicate(server_model, weight_list):
    def __init__(self):
        self.

class ServerTrain:
    def __init__(self, args, args_train, algorithm):
        self.args = args
        self.args_train = args_train
        self.algorithm = algorithm

    # @property
    def train(self):
        client_model = deepcopy(self.args_train['client_model'])
        client_model_ = deepcopy(client_model)
        if self.algorithm == 1:
            # 仅在本地数据上进行训练，不进行蒸馏
            for i in self.args_train['client_idx']:
                message = f'   -client {i}'
                self.args_train['log'].info(message)
                dataloader = self.args_train['train_dataloader'][i]
                model, loss, acc = self.normal_train(
                    client_model[i], dataloader)
                self.args_train['client_loss'][i].extend(loss)
                self.args_train['client_accuracy'][i].extend(acc)
                client_model_[i] = deepcopy(model)
        elif self.algorithm == 2:
            # 仅在本地数据和客户端上训练，不进行蒸馏
            for i in self.args_train['client_idx']:
                message = f'   -client {i}'
                self.args_train['log'].info(message)
                self.args_train['log'].info(message)
                dataloader = self.args_train['train_dataloader'][i]
                model, loss, acc = self.normal_train(
                    client_model[i], dataloader)
                self.args_train['client_loss'][i].extend(loss)
                self.args_train['client_accuracy'][i].extend(acc)
                model, loss, acc = self.normal_train(
                    model, self.args_train['public_dataloader'])
                # client_model.append(model)
                self.args_train['client_loss'][i].extend(loss)
                self.args_train['client_accuracy'][i].extend(acc)
                client_model_[i] = deepcopy(model)
        elif self.algorithm == 3:
            # 本地训练+加权蒸馏
            n = len(self.args_train['server_model'])
            weight = torch.tensor([1/n for _ in range(n)])
            for i in self.args_train['client_idx']:
                message = f'   -client {i}'
                self.args_train['log'].info(message)
                dataloader = self.args_train['train_dataloader'][i]
                model, loss, acc = self.normal_train(
                    client_model[i], dataloader)
                self.args_train['client_loss'][i].extend(loss)
                self.args_train['client_accuracy'][i].extend(acc)
                model, loss, acc = self.weighted_distill_train(model, weight)
                self.args_train['client_loss'][i].extend(loss)
                self.args_train['client_accuracy'][i].extend(acc)
                client_model_[i] = deepcopy(model)
        elif self.algorithm == 4:
            # 本地训练+逐个蒸馏
            for i in self.args_train['client_idx']:
                message = f'   -client {i}'
                self.args_train['log'].info(message)
                dataloader = self.args_train['train_dataloader'][i]
                model, loss, acc = self.normal_train(
                    client_model[i], dataloader)
                self.args_train['client_loss'][i].extend(loss)
                self.args_train['client_accuracy'][i].extend(acc)
                model, loss, acc = self.single_distill_train(model)
                self.args_train['client_loss'][i].extend(loss)
                self.args_train['client_accuracy'][i].extend(acc)
                client_model_[i] = deepcopy(model)
        elif self.algorithm == 5:
            # 交换参数
            for i in self.args_train['client_idx']:
                message = f'   -client {i}'
                self.args_train['log'].info(message)
                dataloader = self.args_train['train_dataloader'][i]
                model, loss, acc = self.normal_train(
                    client_model[i], dataloader)
                self.args_train['client_loss'][i].extend(loss)
                self.args_train['client_accuracy'][i].extend(acc)
                model, loss, acc = self.single_distill_train(model)
                self.args_train['client_loss'][i].extend(loss)
                self.args_train['client_accuracy'][i].extend(acc)
                client_model_[i] = deepcopy(model)
        else:
            raise ValueError('algorithm error.')
        return client_model_

    def normal_train(self, model, dataloader):
        loss_ = []
        acc_ = []
        for epoch in range(self.args.num_client_train):
            model, loss = self.train_model(
                model=model,
                dataloader=dataloader)
            acc = eval_model(
                model=model,
                dataloader=self.args_train['test_dataloader'],
                device=self.args.device).item()
            acc_.append(acc)
            message = '    |{:^15}: {}, acc {:.3f}'.format(
                'local epoch', epoch, acc)
            self.args_train['log'].info(message)
            loss_.extend(loss)
        return model, loss_, acc_

    def weighted_distill_train(self, model, weight):
        loss_ = []
        acc_ = []
        for epoch in range(self.args.num_public_train):
            model, loss = self.train_model_disti_weighted(
                model=model,
                weight=weight)
            loss_.extend(loss)
            acc = eval_model(
                model=model,
                dataloader=self.args_train['test_dataloader'],
                device=self.args.device).item()
            acc_.append(acc)
            message = '    |{:^15}: {}, acc {:.3f}'.format(
                'distill epoch', epoch, acc)
            self.args_train['log'].info(message)
        return model, loss_, acc_

    def single_distill_train(self, model):
        loss_ = []
        acc_ = []
        for epoch in range(self.args.num_public_train):
            for model_ in self.args_train['neighbor']:
                model, loss, acc = self.train_model_disti_single(
                    model=model,
                    teacher_model=model_)
                loss_.append(loss)
                acc = eval_model(
                    model=model,
                    dataloader=self.args_train['test_dataloader'],
                    device=self.args.device).item()
            message = '    |{:^15}: {}, acc {:.3f}'.format(
                'distill epoch', epoch, acc)
            self.args_train['log'].info(message)
            acc_.append(acc)
        return model, loss_, acc_

    # 训练模型
    def train_model(self, model, dataloader):
        device = self.args.device
        trained_model = deepcopy(model).to(device)
        trained_model.train()
        optimizer = torch.optim.Adam(trained_model.parameters())
        loss_ = []
        for data, target in dataloader:
            optimizer.zero_grad()
            output = trained_model(data.to(device))
            loss = f.cross_entropy(output, target.to(device))
            loss.backward()
            optimizer.step()
            loss_.append(loss.item())
        return trained_model, loss_

    # 训练蒸馏模型, logits加权聚合
    def train_model_disti_weighted(self, model, weight):
        device = self.args.device
        trained_model = deepcopy(model).to(device)
        trained_model.train()
        weight_device = weight.to(device)
        optimizer = torch.optim.Adam(trained_model.parameters())
        criterion = DistillKL(T=self.args.T, alpha=self.args.alpha)
        loss_ = []
        for data, target in self.args_train['public_dataloader']:
            data_device = data.to(device)
            teacher_logits = torch.zeros(
                [len(target), self.args_train['num_target']], device=device)
            for i, model in enumerate(self.args_train['neighbor']):
                teacher_model = deepcopy(model).to(device)
                teacher_model.eval()
                teacher_logits += teacher_model(data_device) * weight_device[i]
            optimizer.zero_grad()
            output = trained_model(data_device)
            loss = criterion(output, target.to(device), teacher_logits)
            loss.backward()
            optimizer.step()
            loss_.append(loss.item())
        return trained_model, loss_

    # 训练蒸馏模型, 单个teacher
    def train_model_disti_single(self, model, teacher_model):
        device = self.args.device
        trained_model = deepcopy(model).to(device)
        trained_model.train()
        teacher_model = deepcopy(teacher_model).to(device)
        teacher_model.eval()
        criterion = DistillKL(T=self.args.T, alpha=self.args.alpha)
        optimizer = torch.optim.Adam(trained_model.parameters())
        loss_ = []
        for data, target in self.args_train['public_dataloader']:
            optimizer.zero_grad()
            logits = teacher_model(data.to(device))
            output = trained_model(data.to(device))
            loss = criterion(output, target.to(device), logits)
            loss.backward()
            optimizer.step()
            loss_.append(loss.item())
        return trained_model, loss_


def aggregate(model_list, weight):
    aggregated_model = deepcopy(model_list[0])
    parameters = deepcopy(model_list[0].state_dict())
    for key in parameters:
        parameters[key] *= weight[0]
    for i, model in enumerate(model_list[1:]):
        for key in parameters:
            parameters[key] += model.state_dict()[key] * weight[i+1]
    aggregated_model.load_state_dict(parameters)
    return aggregated_model


# class EdgeServer:
#     '''
#     边缘服务器聚合
#     '''

#     def __init__(self, client_model):
#         self.model = deepcopy(client_model[0])
#         self.num_client = len(client_model)

#         self.client_params = list_same_term(self.num_client)
#         i = 0
#         for client in client_model:
#             self.client_params[i] = client.state_dict()
#             i += 1

#     # 平均
#     def average(self):
#         model = deepcopy(self.model)
#         parameters = deepcopy(self.client_params[0])
#         for client in range(1, self.num_client):
#             for key in parameters:
#                 parameters[key] += self.client_params[client][key]
#         for key in parameters:
#             parameters[key] /= self.num_client
#         model.load_state_dict(parameters)
#         return model

#     # 加权平均
#     def weighted_average(self, weight):
#         model = deepcopy(self.model)
#         parameters = deepcopy(self.client_params[0])
#         for key in parameters:
#             parameters[key] *= weight[0]
#         for client in range(1, self.num_client):
#             for key in parameters:
#                 parameters[key] += self.client_params[client][key] * \
#                     weight[client]
#         model.load_state_dict(parameters)
#         return model


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
