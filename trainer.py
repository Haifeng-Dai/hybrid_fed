import torch

from utils.model_util import *
from utils.data_util import *
from utils.lib_util import *
from utils.train_util import *


# class ServerTrain:
#     def __init__(self, args, args_train, train_way):
#         self.args = args
#         self.args_train = args_train
#         self.train_way = train_way
#         self.LR = self.args_train['LR']

#     @property
#     def train(self):
#         client_model = deepcopy(self.args_train['client_model'])
#         client_model_ = deepcopy(client_model)
#         if self.train_way == 1:
#             # 仅在本地数据上进行训练，不进行蒸馏
#             for i in self.args_train['client_idx']:
#                 message = f' ---client {i}'
#                 self.args_train['log'].info(message)
#                 dataloader = self.args_train['train_dataloader'][i]
#                 test_dataloader = self.args_train['train_test_dataloader'][i]
#                 model, loss, acc, acc_train = self.normal_train(
#                     client_model[i], dataloader, test_dataloader)
#                 self.args_train['client_loss'][i].extend(loss)
#                 self.args_train['client_accuracy'][i].extend(acc)
#                 self.args_train['train_accuracy'][i].extend(acc_train)
#                 client_model_[i] = deepcopy(model)
#         elif self.train_way == 2:
#             # 仅在本地数据和公开数据集上训练，不进行蒸馏
#             for i in self.args_train['client_idx']:
#                 message = f' ---client {i}'
#                 self.args_train['log'].info(message)
#                 self.args_train['log'].info(message)
#                 dataloader = self.args_train['train_dataloader'][i]
#                 test_dataloader = self.args_train['train_test_dataloader'][i]
#                 model, loss, acc, acc_train = self.normal_train(
#                     client_model[i], dataloader, test_dataloader)
#                 self.args_train['client_loss'][i].extend(loss)
#                 self.args_train['client_accuracy'][i].extend(acc)
#                 model, loss, acc, acc_train = self.normal_train(
#                     model, self.args_train['public_dataloader'], test_dataloader)
#                 # client_model.append(model)
#                 self.args_train['client_loss'][i].extend(loss)
#                 self.args_train['client_accuracy'][i].extend(acc)
#                 self.args_train['train_accuracy'][i].extend(acc_train)
#                 client_model_[i] = deepcopy(model)
#         elif self.train_way == 3:
#             # 本地训练+加权蒸馏
#             weight = torch.tensor([1/3, 1/3, 1/3])
#             for i in self.args_train['client_idx']:
#                 message = f' ---client {i}'
#                 self.args_train['log'].info(message)
#                 dataloader = self.args_train['train_dataloader'][i]
#                 test_dataloader = self.args_train['train_test_dataloader'][i]
#                 model, loss, acc, acc_train = self.normal_train(
#                     client_model[i], dataloader, test_dataloader)
#                 self.args_train['client_loss'][i].extend(loss)
#                 self.args_train['client_accuracy'][i].extend(acc)
#                 self.args_train['train_accuracy'][i].extend(acc_train)
#                 model, loss, acc, acc_train = self.weighted_distill_train(
#                     model, weight, test_dataloader)
#                 self.args_train['client_loss'][i].extend(loss)
#                 self.args_train['client_accuracy'][i].extend(acc)
#                 self.args_train['train_accuracy'][i].extend(acc_train)
#                 client_model_[i] = deepcopy(model)
#         elif self.train_way == 4:
#             # 本地训练+逐个蒸馏
#             for i in self.args_train['client_idx']:
#                 message = f' ---client {i}'
#                 self.args_train['log'].info(message)
#                 dataloader = self.args_train['train_dataloader'][i]
#                 test_dataloader = self.args_train['train_test_dataloader'][i]
#                 model, loss, acc, acc_train = self.normal_train(
#                     client_model[i], dataloader, test_dataloader)
#                 self.args_train['client_loss'][i].extend(loss)
#                 self.args_train['client_accuracy'][i].extend(acc)
#                 self.args_train['train_accuracy'][i].extend(acc_train)
#                 model, loss, acc, acc_train = self.single_distill_train(
#                     model, test_dataloader)
#                 self.args_train['client_loss'][i].extend(loss)
#                 self.args_train['client_accuracy'][i].extend(acc)
#                 self.args_train['train_accuracy'][i].extend(acc_train)
#                 client_model_[i] = deepcopy(model)
#         else:
#             raise ValueError('algorithm error.')
#         return client_model_


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
        model, loss, acc, acc_val = regular_loop(
            client_model[i], dataloader, validate_dataloader, args, args_train)
        args_train['client_loss'][i].extend(loss)
        args_train['client_accuracy'][i].extend(acc)
        args_train['train_accuracy'][i].extend(acc_val)
        if args_train['epoch_server_commu'] != 0:
            model, loss, acc, acc_val = weighted_distill_train_loop(
                model, weight, validate_dataloader, args, args_train)
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
        model, loss, acc, acc_val = regular_loop(
            client_model[i], dataloader, validate_dataloader, args, args_train)
        args_train['client_loss'][i].extend(loss)
        args_train['client_accuracy'][i].extend(acc)
        args_train['train_accuracy'][i].extend(acc_val)
        if args_train['epoch_server_commu'] != 0:
            model, loss, acc, acc_val = circulate_distill_train_loop(
                model, validate_dataloader, args, args_train)
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
        dataloader = args_train['train_dataloader'][i]
        validate_dataloader = args_train['validate_dataloader'][i]
        model, loss, acc, acc_val = regular_loop(
            client_model[i], dataloader, validate_dataloader, args, args_train)
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
    args_train['server_model'][server] = aggregate(
        server_model_, args_train['weight_server'])
    for client in args_train['server_client'][server]:
        args_train['client_model'][client] = deepcopy(
            args_train['server_model'][server])
    # 评估单个服务器模型
    acc_server = eval_model(
        model=args_train['server_model'][server],
        dataloader=args_train['test_dataloader'],
        device=args.device)
    return acc_server


def server_communicate(args, args_train):
    if args.algorithm != 2:
        return
    server_model_ = deepcopy(args_train['server_model'])
    for i, weight in enumerate(args_train['weight_list']):
        server_model_[i] = aggregate(args_train['server_model'], weight)
    for server in args_train['all_server']:
        for client in args_train['server_client'][server]:
            args_train['client_model'][client] = deepcopy(
                args_train['server_model'][server])
    args_train['server_model'] = deepcopy(server_model_)


class Trainer:
    def __init__(self, neighbor_server, args, args_train):
        self.neighbor_server = neighbor_server
        self.args = args
        self.args_train = args_train
        if args.algorithm == 0:
            # args_train['epoch_server_commu'] = epoch_server_commu
            self.trainer = weighted_distill
        elif args.algorithm == 1:
            # args_train['epoch_server_commu'] = epoch_server_commu
            self.trainer = circulate_distill
        elif args.algorithm == 2:
            self.trainer = regular
        elif args.algorithm == 3:
            self.trainer == regular
        else:
            raise ValueError('algorithm error.')

    # @property
    # def train(self):
    #     return self.trainer(self.args, self.args_train)

    @property
    def train(self):
        # def trainer(neighbor_server, args, args_train):
        # neighbor_server = self.neighbor_server
        # args = self.args
        # args_train = self.args_train
        # all_server = [i for i in range(self.args.num_all_server)]
        server_accuracy = list_same_term(self.args.num_all_server)
        d = 2
        for epoch_server_commu in range(self.args.num_server_commu):
            self.args_train['epoch_server_commu'] = epoch_server_commu
            self.args_train['log'].info('-'*50)
            self.args_train['log'].info('|epoch_server_commu: {}/{}'.format(
                epoch_server_commu, self.args.num_server_commu))
            self.args_train['LR'] = 1e-3 / (1 + d * epoch_server_commu)
            # 所有边缘服务器分别协调其客户端进行多轮联邦学习
            for epoch_client_commu in range(self.args.num_client_commu):
                message = ' |epoch_client_commu: {}/{}'.format(
                    epoch_client_commu, self.args.num_client_commu)
                self.args_train['log'].info(message)
                # 所有边缘服务器分别协调其客户端进行联邦学习
                neighbor_model = []
                for server in self.args_train['all_server']:
                    # 每个服务器下单客户端分别训练
                    message = f'  |server: {server}'
                    self.args_train['log'].info(message)
                    self.args_train['client_idx'] = self.args_train['server_client'][server]
                    for i in self.neighbor_server[server]:
                        neighbor_model.append(
                            self.args_train['server_model'][i])
                    self.args_train['neighbor'] = neighbor_model
                    client_model = self.trainer(self.args, self.args_train)
                    self.args_train['client_model'] = deepcopy(client_model)
                    acc_server = aggregator(server, self.args, self.args_train)
                    message = '|servers comunicated: {}, server aggregated: {}, acc_server {}: {:.3f}.'.format(
                        epoch_server_commu, epoch_client_commu, server, acc_server)
                    self.args_train['log'].info(message)
                    self.args_train['log'].info('-'*50)
                    server_accuracy[server].append(acc_server)
                server_communicate(self.args, self.args_train)
            message = '{:^50}'.format(
                '********  servers comunicates  ********')
            self.args_train['log'].info(message)
        return server_accuracy
