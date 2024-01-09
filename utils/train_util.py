import torch
from copy import deepcopy


def ce_loss(output, target, device):
    return torch.nn.CrossEntropyLoss().to(device)(output, target)


def kl_loss(output, target, device):
    return torch.nn.KLDivLoss(reduction='batchmean').to(device)(output, target)


def softmax_loss(tensor, device):
    return torch.nn.Softmax(dim=1).to(device)(tensor)


class DistillKL(torch.nn.Module):
    '''
    distilling loss
    '''

    def __init__(self, T, alpha, device):
        super(DistillKL, self).__init__()
        self.T = T
        self.alpha = alpha
        self.device = device

    def forward(self, output, target, logits_teacher):
        prob_teacher = softmax_loss(logits_teacher/self.T, self.device)
        prob_student = softmax_loss(output/self.T, self.device)
        soft_loss = kl_loss(prob_student.log(), prob_teacher, self.device)
        hard_loss = ce_loss(output, target, self.device)
        loss = self.alpha * hard_loss + \
            (1 - self.alpha) * soft_loss * self.T**2 / logits_teacher.shape[0]
        return loss


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
        loss = ce_loss(output, target.to(device), device)
        loss.backward()
        optimizer.step()
        loss_.append(loss.item())
    return trained_model, loss_


def regular_loop(model, dataloader, validate_dataloader, args, args_train):
    # 不进行蒸馏训练
    loss_ = []
    acc_ = []
    acc__ = []
    for epoch in range(args.num_client_train):
        model, loss = train_model(
            model=model,
            dataloader=dataloader,
            device=args.device,
            LR=args_train['LR'])
        acc = eval_model(
            model=model,
            dataloader=args_train['test_dataloader'],
            device=args.device)
        acc_val = eval_model(
            model=model,
            dataloader=validate_dataloader,
            device=args.device)
        acc_.append(acc)
        acc__.append(acc_val)
        message = '|{:^15}: {}, acc {:.3f}'.format(
            'local epoch', epoch, acc)
        args_train['log'].info(message)
        loss_.extend(loss)
    return model, loss_, acc_, acc__


def weighted_distill_train_loop(model, weight, validate_dataloader, args, args_train):
    # 训练蒸馏模型, logits加权聚合
    loss_ = []
    acc_ = []
    acc__ = []
    for epoch in range(args.num_public_train):
        trained_model = deepcopy(model).to(args.device).train()
        weight_device = weight.to(args.device)
        optimizer = torch.optim.Adam(trained_model.parameters(),
                                     lr=args_train['LR'],
                                     weight_decay=1e-3)
        criterion = DistillKL(args.T, args.alpha, args.device)
        for data, target in args_train['public_dataloader']:
            data_device = data.to(args.device)
            teacher_logits = torch.zeros(
                [len(target), args_train['num_target']], device=args.device)
            for i, model_ in enumerate(args_train['neighbor']):
                teacher_model = deepcopy(model_).to(args.device)
                teacher_model.eval()
                teacher_logits += teacher_model(data_device) * weight_device[i]
            optimizer.zero_grad()
            output = trained_model(data_device)
            loss = criterion(output, target.to(args.device), teacher_logits)
            loss.backward()
            optimizer.step()
            loss_.append(loss.item())
        acc = eval_model(
            model=trained_model,
            dataloader=args_train['test_dataloader'],
            device=args.device)
        acc_val = eval_model(
            model=trained_model,
            dataloader=validate_dataloader,
            device=args.device)
        acc_.append(acc)
        acc__.append(acc_val)
        message = '|{:^15}: {}, acc {:.3f}'.format(
            'distill epoch', epoch, acc)
        args_train['log'].info(message)
    return trained_model, loss_, acc_, acc__


def circulate_distill_train_loop(model, validate_dataloader, args, args_train):
    # 训练蒸馏模型, 每个teacher循环
    loss_ = []
    acc_ = []
    acc__ = []
    for epoch in range(args.num_public_train):
        _acc = []
        __acc = []
        for model_ in args_train['neighbor']:
            trained_model = deepcopy(model).to(args.device)
            trained_model.train()
            teacher_model = deepcopy(model_).to(args.device)
            teacher_model.eval()
            criterion = DistillKL(args.T, args.alpha, args.device)
            optimizer = torch.optim.Adam(trained_model.parameters(),
                                         lr=args_train['LR'],
                                         weight_decay=1e-3)
            for data, target in args_train['public_dataloader']:
                optimizer.zero_grad()
                logits = teacher_model(data.to(args.device))
                output = trained_model(data.to(args.device))
                loss = criterion(output, target.to(args.device), logits)
                loss.backward()
                optimizer.step()
                loss_.append(loss.item())
            acc = eval_model(
                model=trained_model,
                dataloader=args_train['test_dataloader'],
                device=args.device)
            acc_val = eval_model(
                model=trained_model,
                dataloader=validate_dataloader,
                device=args.device)
            message = '|{:^15}: {}, acc {:.3f}'.format(
                'distill epoch', epoch, acc)
            args_train['log'].info(message)
            _acc.append(acc)
            __acc.append(acc_val)
        acc_.extend(_acc)
        acc__.extend(__acc)
    return trained_model, loss_, acc_, acc__

# def train_model_weighted_distill(model, weight, alpha, T, dataloader, num_target, neighbor, device, LR):
#     # 训练蒸馏模型, logits加权聚合
#     trained_model = deepcopy(model).to(device)
#     trained_model.train()
#     weight_device = weight.to(device)
#     optimizer = torch.optim.Adam(trained_model.parameters(),
#                                  lr=LR,
#                                  weight_decay=1e-3)
#     criterion = DistillKL(T, alpha, device)
#     loss_ = []
#     for data, target in dataloader:
#         data_device = data.to(device)
#         teacher_logits = torch.zeros(
#             [len(target), num_target], device=device)
#         for i, model in enumerate(neighbor):
#             teacher_model = deepcopy(model).to(device)
#             teacher_model.eval()
#             teacher_logits += teacher_model(data_device) * weight_device[i]
#         optimizer.zero_grad()
#         output = trained_model(data_device)
#         loss = criterion(output, target.to(device), teacher_logits)
#         loss.backward()
#         optimizer.step()
#         loss_.append(loss.item())
#     return trained_model, loss_


# def train_model_single_distill(model, teacher_model, dataloader, alpha, T, device, LR):
#     # 训练蒸馏模型, 单个teacher
#     trained_model = deepcopy(model).to(device)
#     trained_model.train()
#     teacher_model = deepcopy(teacher_model).to(device)
#     teacher_model.eval()
#     criterion = DistillKL(T, alpha, device)
#     optimizer = torch.optim.Adam(trained_model.parameters(),
#                                  lr=LR,
#                                  weight_decay=1e-3)
#     loss_ = []
#     for data, target in dataloader:
#         optimizer.zero_grad()
#         logits = teacher_model(data.to(device))
#         output = trained_model(data.to(device))
#         loss = criterion(output, target.to(device), logits)
#         loss.backward()
#         optimizer.step()
#         loss_.append(loss.item())
#     return trained_model, loss_


# def aggregate(model_list, weight):
#     aggregated_model = deepcopy(model_list[0])
#     parameters = deepcopy(model_list[0].state_dict())
#     for key in parameters:
#         print(type(parameters[key]), type(weight[0]))
#         parameters[key] *= weight[0]
#     for i, model in enumerate(model_list[1:]):
#         for key in parameters:
#             parameters[key] += model.state_dict()[key] * weight[i+1]
#     aggregated_model.load_state_dict(parameters)
#     return aggregated_model


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
