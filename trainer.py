import torch

from utils.model_util import *
from utils.data_util import *
from utils.lib_util import *
from utils.train_util import *


def weighted_distill(args, args_train):
    # 本地训练+加权蒸馏
    client_model = deepcopy(args_train['client_model'])
    client_model_ = deepcopy(client_model)
    weight = torch.tensor([1/3, 1/3, 1/3])
    for i in args_train['client_idx']:
        message = f' ---client {i}'
        args_train['log'].info(message)
        dataloader = args_train['train_dataloader'][i]
        validate_dataloader = args_train['validate_dataloader'][i]
        model, loss, acc, acc_val = regular_loop(model=client_model[i],
                                                 dataloader=dataloader,
                                                 validate_dataloader=validate_dataloader,
                                                 args=args,
                                                 args_train=args_train)
        args_train['client_loss'][i].extend(loss)
        args_train['client_accuracy'][i].extend(acc)
        args_train['validate_accuracy'][i].extend(acc_val)
        if args_train['epoch_server_commu'] != 0:
            model, loss, acc, acc_val = weighted_distill_train_loop(model=model,
                                                                    weight=weight,
                                                                    validate_dataloader=validate_dataloader,
                                                                    args=args,
                                                                    args_train=args_train)
            args_train['client_loss'][i].extend(loss)
            args_train['client_accuracy'][i].extend(acc)
            args_train['validate_accuracy'][i].extend(acc_val)
        client_model_[i] = deepcopy(model)
    return client_model_


def circulate_distill(args, args_train):
    # 本地训练+循环蒸馏
    client_model = deepcopy(args_train['client_model'])
    client_model_ = deepcopy(client_model)
    for i in args_train['client_idx']:
        message = f' ---client {i}'
        args_train['log'].info(message)
        dataloader = args_train['train_dataloader'][i]
        validate_dataloader = args_train['validate_dataloader'][i]
        model, loss, acc, acc_val = regular_loop(model=client_model[i],
                                                 dataloader=dataloader,
                                                 validate_dataloader=validate_dataloader,
                                                 args=args,
                                                 args_train=args_train)
        args_train['client_loss'][i].extend(loss)
        args_train['client_accuracy'][i].extend(acc)
        args_train['validate_accuracy'][i].extend(acc_val)
        if args_train['epoch_server_commu'] != 0:
            model, loss, acc, acc_val = circulate_distill_train_loop(model=model,
                                                                     validate_dataloader=validate_dataloader,
                                                                     args=args,
                                                                     args_train=args_train)
            args_train['client_loss'][i].extend(loss)
            args_train['client_accuracy'][i].extend(acc)
            args_train['validate_accuracy'][i].extend(acc_val)
        client_model_[i] = deepcopy(model)
    return client_model_


def regular(args, args_train):
    # 不进行蒸馏
    client_model = deepcopy(args_train['client_model'])
    client_model_ = deepcopy(client_model)
    for i in args_train['client_idx']:
        message = f' ---client {i}'
        args_train['log'].info(message)
        dataloader = args_train['regular_dataloader'][i]
        validate_dataloader = args_train['validate_dataloader'][i]
        model, loss, acc, acc_val = regular_loop(model=client_model[i],
                                                 dataloader=dataloader,
                                                 validate_dataloader=validate_dataloader,
                                                 args=args,
                                                 args_train=args_train)
        client_model_[i] = deepcopy(model)
        args_train['client_loss'][i].extend(loss)
        args_train['client_accuracy'][i].extend(acc)
        args_train['validate_accuracy'][i].extend(acc_val)
    return client_model_


def aggregator(server, args, args_train):
    # 在单个服务器下客户端训练完成后更新该服务器下客户端的模型
    server_model_ = [
        args_train['client_model'][client] for client in args_train['server_client'][server]]
    # 聚合获得单个服务器模型并下发
    args_train['server_model'][server] = aggregate(model_list=server_model_,
                                                   weight=args_train['weight_server'])
    for client in args_train['server_client'][server]:
        args_train['client_model'][client] = deepcopy(
            args_train['server_model'][server])
    # 评估单个服务器模型
    acc_server = eval_model(model=args_train['server_model'][server],
                            dataloader=args_train['test_dataloader'],
                            device=args.device)
    return acc_server


def server_communicate(args_train):
    server_model_ = deepcopy(args_train['server_model'])
    client_model = deepcopy(args_train['client_model'])
    for i, weight in enumerate(args_train['weight_list']):
        server_model_[i] = aggregate(model_list=args_train['server_model'],
                                     weight=weight)
    # for server in args_train['all_server']:
        for client in args_train['server_client'][i]:
            client_model[client] = deepcopy(server_model_[i])
            # args_train['client_model'][client] = deepcopy(
            #     args_train['server_model'][server])
    # args_train['server_model'] = deepcopy(server_model_)
        # for client in args_train['server_client'][server]:
        #     client_model[client] = server_model_[server]
    return server_model_, client_model


def server_distill(args, args_train):
    # 本地训练+服务器蒸馏
    weight_device = torch.tensor([1/3, 1/3, 1/3], device=args.device)
    server_model = deepcopy(args_train['server_model'])
    client_model = deepcopy(args_train['client_model'])
    server_model_ = {}
    message = f'|server distill'
    args_train['log'].info(message)
    msg = '|server: {}, acc: {:.3f}'
    for server in args_train['all_server']:
        trained_model = server_model[server]
        for i in range(args.num_public_train):
            optimizer = torch.optim.Adam(params=trained_model.parameters(),
                                         lr=args_train['LR'],
                                         weight_decay=1e-3)
            criterion = DistillKL(T=args.T,
                                  alpha=args.alpha,
                                  device=args.device)
            for data, target in args_train['public_dataloader']:
                data_device = data.to(args.device)
                logits = torch.zeros(
                    [len(target), args_train['num_target']], device=args.device)
                for i, model_ in enumerate(args_train['neighbor']):
                    teacher_model = deepcopy(model_).to(args.device)
                    teacher_model.eval()
                    logits += teacher_model(data_device) * \
                        weight_device[i]
                optimizer.zero_grad()
                output = trained_model(data_device)
                loss = criterion(output, target.to(args.device), logits)
                loss.backward()
                optimizer.step()
            acc = eval_model(model=trained_model,
                             dataloader=args_train['test_dataloader'],
                             device=args.device)
            message = msg.format(server, acc)
            args_train['log'].info(message)
        server_model_[server] = deepcopy(trained_model)
        for client in args_train['server_client'][server]:
            client_model[client] = deepcopy(trained_model)
    return server_model_, client_model


class Trainer:
    def __init__(self, neighbor_server, args, args_train):
        self.neighbor_server = neighbor_server
        self.args = args
        self.args_train = args_train
        if self.args.algorithm == 0:
            self.trainer = weighted_distill
        elif self.args.algorithm == 1:
            self.trainer = circulate_distill
        elif self.args.algorithm == 2 or self.args.algorithm == 3:
            self.trainer = regular
        else:
            raise ValueError('algorithm error.')

    @property
    def train(self):
        msg1 = ' |epoch_client_commu: {}/{}'
        msg2 = '|servers comunicated: {}, server aggregated: {}, acc_server {}: {:.3f}.'
        server_accuracy = list_same_term(self.args.num_all_server)
        d = 2
        for epoch_server_commu in range(self.args.num_server_commu):
            self.args_train['epoch_server_commu'] = epoch_server_commu
            self.args_train['log'].info('-'*50)
            self.args_train['log'].info('|epoch_server_commu: {}/{}'.format(
                epoch_server_commu, self.args.num_server_commu))
            self.args_train['LR'] = 1e-3 / (1 + d * epoch_server_commu)
            # FedAvg
            for epoch_client_commu in range(self.args.num_client_commu):
                message = msg1.format(epoch_client_commu,
                                      self.args.num_client_commu)
                self.args_train['log'].info(message)
                # 所有边缘服务器分别协调其客户端进行联邦学习
                neighbor_model = []
                for server in self.args_train['all_server']:
                    # 每个服务器下单客户端分别训练
                    self.args_train['log'].info(f'  |server: {server}')
                    self.args_train['client_idx'] = self.args_train['server_client'][server]
                    for i in self.neighbor_server[server]:
                        neighbor_model.append(
                            self.args_train['server_model'][i])
                    self.args_train['neighbor'] = neighbor_model
                    client_model = self.trainer(self.args, self.args_train)
                    self.args_train['client_model'] = deepcopy(client_model)
                    acc_server = aggregator(server=server,
                                            args=self.args,
                                            args_train=self.args_train)
                    message = msg2.format(epoch_server_commu,
                                          epoch_client_commu, server,
                                          acc_server)
                    self.args_train['log'].info(message)
                    self.args_train['log'].info('-'*50)
                    server_accuracy[server].append(acc_server)
                # server distill
                if self.args.algorithm == 3:
                    if epoch_server_commu == 0:
                        continue
                    server_client_model = server_distill(args=self.args,
                                                        args_train=self.args_train)
                    self.args_train['server_model'] = deepcopy(
                        server_client_model[0])
                    self.args_train['client_model'] = deepcopy(
                        server_client_model[1])
            # server communicate
            if self.args.algorithm == 2:
                server_client_model = server_communicate(self.args_train)
                self.args_train['server_model'] = deepcopy(
                    server_client_model[0])
                self.args_train['client_model'] = deepcopy(
                    server_client_model[1])
            message = '{:^50}'.format(
                '********  servers comunicates  ********')
            self.args_train['log'].info(message)
        return server_accuracy
