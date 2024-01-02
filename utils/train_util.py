import torch

from copy import deepcopy
import torch.nn.functional as f

# from utils.lib_util import list_same_term


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
        soft_loss = f.kl_div(prob_student.log(),
                             prob_teacher, reduction='batchmean')
        hard_loss = f.cross_entropy(output, target)
        loss = self.alpha * hard_loss + \
            (1 - self.alpha) * soft_loss * self.T**2 / logits_teacher.shape[0]
        return loss


class ServerTrain:
    def __init__(self, args, args_train, train_way):
        self.args = args
        self.args_train = args_train
        self.train_way = train_way
        self.LR = self.args_train['LR']

    @property
    def train(self):
        client_model = deepcopy(self.args_train['client_model'])
        client_model_ = deepcopy(client_model)
        if self.train_way == 1:
            # 仅在本地数据上进行训练，不进行蒸馏
            for i in self.args_train['client_idx']:
                message = f'   -client {i}'
                self.args_train['log'].info(message)
                dataloader = self.args_train['train_dataloader'][i]
                test_dataloader = self.args_train['train_test_dataloader'][i]
                model, loss, acc, acc_train = self.normal_train(
                    client_model[i], dataloader, test_dataloader)
                self.args_train['client_loss'][i].extend(loss)
                self.args_train['client_accuracy'][i].extend(acc)
                self.args_train['train_accuracy'][i].extend(acc_train)
                client_model_[i] = deepcopy(model)
        elif self.train_way == 2:
            # 仅在本地数据和公开数据集上训练，不进行蒸馏
            for i in self.args_train['client_idx']:
                message = f'   -client {i}'
                self.args_train['log'].info(message)
                self.args_train['log'].info(message)
                dataloader = self.args_train['train_dataloader'][i]
                test_dataloader = self.args_train['train_test_dataloader'][i]
                model, loss, acc, acc_train = self.normal_train(
                    client_model[i], dataloader, test_dataloader)
                self.args_train['client_loss'][i].extend(loss)
                self.args_train['client_accuracy'][i].extend(acc)
                model, loss, acc, acc_train = self.normal_train(
                    model, self.args_train['public_dataloader'], test_dataloader)
                # client_model.append(model)
                self.args_train['client_loss'][i].extend(loss)
                self.args_train['client_accuracy'][i].extend(acc)
                self.args_train['train_accuracy'][i].extend(acc_train)
                client_model_[i] = deepcopy(model)
        elif self.train_way == 3:
            # 本地训练+加权蒸馏
            # n = len(self.args_train['server_model'])
            weight = torch.tensor([1/3, 1/3, 1/3])
            for i in self.args_train['client_idx']:
                message = f'   -client {i}'
                self.args_train['log'].info(message)
                dataloader = self.args_train['train_dataloader'][i]
                test_dataloader = self.args_train['train_test_dataloader'][i]
                model, loss, acc, acc_train = self.normal_train(
                    client_model[i], dataloader, test_dataloader)
                self.args_train['client_loss'][i].extend(loss)
                self.args_train['client_accuracy'][i].extend(acc)
                self.args_train['train_accuracy'][i].extend(acc_train)
                model, loss, acc, acc_train = self.weighted_distill_train(
                    model, weight, test_dataloader)
                self.args_train['client_loss'][i].extend(loss)
                self.args_train['client_accuracy'][i].extend(acc)
                self.args_train['train_accuracy'][i].extend(acc_train)
                client_model_[i] = deepcopy(model)
        elif self.train_way == 4:
            # 本地训练+逐个蒸馏
            for i in self.args_train['client_idx']:
                message = f'   -client {i}'
                self.args_train['log'].info(message)
                dataloader = self.args_train['train_dataloader'][i]
                test_dataloader = self.args_train['train_test_dataloader'][i]
                model, loss, acc, acc_train = self.normal_train(
                    client_model[i], dataloader, test_dataloader)
                self.args_train['client_loss'][i].extend(loss)
                self.args_train['client_accuracy'][i].extend(acc)
                self.args_train['train_accuracy'][i].extend(acc_train)
                model, loss, acc, acc_train = self.single_distill_train(
                    model, test_dataloader)
                self.args_train['client_loss'][i].extend(loss)
                self.args_train['client_accuracy'][i].extend(acc)
                self.args_train['train_accuracy'][i].extend(acc_train)
                client_model_[i] = deepcopy(model)
        # elif self.algorithm == 5:
        #     # 交换参数
        #     for i in self.args_train['client_idx']:
        #         message = f'   -client {i}'
        #         self.args_train['log'].info(message)
        #         dataloader = self.args_train['train_dataloader'][i]
        #         model, loss, acc = self.normal_train(
        #             client_model[i], dataloader)
        #         self.args_train['client_loss'][i].extend(loss)
        #         self.args_train['client_accuracy'][i].extend(acc)
        #         model, loss, acc = self.single_distill_train(model)
        #         self.args_train['client_loss'][i].extend(loss)
        #         self.args_train['client_accuracy'][i].extend(acc)
        #         client_model_[i] = deepcopy(model)
        else:
            raise ValueError('algorithm error.')
        return client_model_

    def normal_train(self, model, dataloader, test_dataloader):
        loss_ = []
        acc_ = []
        acc__ = []
        for epoch in range(self.args.num_client_train):
            model, loss = train_model(
                model=model,
                dataloader=dataloader,
                device=self.args.device,
                LR=self.LR)
            acc = eval_model(
                model=model,
                dataloader=self.args_train['test_dataloader'],
                device=self.args.device)
            acc_train = eval_model(
                model=model,
                dataloader=test_dataloader,
                device=self.args.device)
            acc_.append(acc)
            acc__.append(acc_train)
            message = '    |{:^15}: {}, acc {:.3f}'.format(
                'local epoch', epoch, acc)
            self.args_train['log'].info(message)
            loss_.extend(loss)
        return model, loss_, acc_, acc__

    def weighted_distill_train(self, model, weight, test_dataloader):
        loss_ = []
        acc_ = []
        acc__ = []
        for epoch in range(self.args.num_public_train):
            model, loss = train_model_disti_weighted(
                model=model,
                weight=weight,
                alpha=self.args.alpha,
                T=self.args.T,
                dataloader=self.args_train['public_dataloader'],
                num_target=self.args_train['num_target'],
                neighbor=self.args_train['neighbor'],
                device=self.args.device,
                LR=self.LR)
            loss_.extend(loss)
            acc = eval_model(
                model=model,
                dataloader=self.args_train['test_dataloader'],
                device=self.args.device)
            acc_train = eval_model(
                model=model,
                dataloader=test_dataloader,
                device=self.args.device)
            acc_.append(acc)
            acc__.append(acc_train)
            message = '    |{:^15}: {}, acc {:.3f}'.format(
                'distill epoch', epoch, acc)
            self.args_train['log'].info(message)
        return model, loss_, acc_, acc__

    def single_distill_train(self, model, test_dataloader):
        loss_ = []
        acc_ = []
        acc__ = []
        for epoch in range(self.args.num_public_train):
            _loss = []
            _acc = []
            __acc = []
            for model_ in self.args_train['neighbor']:
                model, loss = train_model_disti_single(
                    model=model,
                    teacher_model=model_,
                    dataloader=self.args_train['public_dataloader'],
                    alpha=self.args.alpha,
                    T=self.args.T,
                    device=self.args.device,
                    LR=self.LR)
                _loss.extend(loss)
                acc = eval_model(
                    model=model,
                    dataloader=self.args_train['test_dataloader'],
                    device=self.args.device)
                acc_train = eval_model(
                    model=model,
                    dataloader=test_dataloader,
                    device=self.args.device)
                message = '    |{:^15}: {}, acc {:.3f}'.format(
                    'distill epoch', epoch, acc)
                self.args_train['log'].info(message)
                _acc.append(acc)
                __acc.append(acc_train)
            loss_.extend(_loss)
            acc_.extend(_acc)
            acc__.extend(__acc)
        return model, loss_, acc_, acc__


def train_model(model, dataloader, device, LR):
    # 训练模型
    trained_model = deepcopy(model).to(device)
    trained_model.train()
    optimizer = torch.optim.Adam(trained_model.parameters(),
                                 lr=LR,
                                 weight_decay=5e-4)
    loss_ = []
    for data, target in dataloader:
        optimizer.zero_grad()
        output = trained_model(data.to(device))
        loss = f.cross_entropy(output, target.to(device))
        loss.backward()
        optimizer.step()
        loss_.append(loss.item())
    return trained_model, loss_


def train_model_disti_weighted(model, weight, alpha, T, dataloader, num_target, neighbor, device, LR):
    # 训练蒸馏模型, logits加权聚合
    trained_model = deepcopy(model).to(device)
    trained_model.train()
    weight_device = weight.to(device)
    optimizer = torch.optim.Adam(trained_model.parameters(),
                                 lr=LR,
                                 weight_decay=1e-3)
    criterion = DistillKL(T, alpha)
    loss_ = []
    for data, target in dataloader:
        data_device = data.to(device)
        teacher_logits = torch.zeros(
            [len(target), num_target], device=device)
        for i, model in enumerate(neighbor):
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


def train_model_disti_single(model, teacher_model, dataloader, alpha, T, device, LR):
    # 训练蒸馏模型, 单个teacher
    trained_model = deepcopy(model).to(device)
    trained_model.train()
    teacher_model = deepcopy(teacher_model).to(device)
    teacher_model.eval()
    criterion = DistillKL(T, alpha)
    optimizer = torch.optim.Adam(trained_model.parameters(),
                                 lr=LR,
                                 weight_decay=1e-3)
    loss_ = []
    for data, target in dataloader:
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


def server_communicate(server_model, weight_list):
    server_model_ = deepcopy(server_model)
    for i, weight in enumerate(weight_list):
        server_model_[i] = aggregate(server_model, weight)
    return server_model_


def eval_model(model, dataloader, device):
    '''
    评估模型
    '''
    model_copy = deepcopy(model).to(device)
    # model_copy.to(device)
    model_copy.eval()
    correct = 0
    len_data = 0
    for images, targets in dataloader:
        outputs = model_copy(images.to(device))
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == targets.to(device)).sum()
        len_data += len(targets)
    accuracy = correct / len_data
    return accuracy.item()
