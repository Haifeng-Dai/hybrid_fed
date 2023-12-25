from copy import deepcopy

from utils.train_util import *
from utils.lib_util import *


class Server:
    def __init__(self, args, args_train, trainer):
        self.args = args
        self.args_train = args_train
        self.trainer = trainer
        # self.model = self.train()

    # @property
    def train(self):
        # client_model = deepcopy(self.args_train['client_model'])
        client_model = self.args_train['client_model']
        if self.trainer == 1:
            # 仅在本地数据上进行训练，不进行蒸馏
            # client_model = []
            # loss_ = list_same_term(self.args_train['num_server_client'])
            # acc_ = deepcopy(loss_)
            for i in self.args_train['client_idx']:
                message = f'   -client {i}'
                self.args_train['log'].info(message)
                model = client_model[i]
                dataloader = self.args_train['train_dataloader'][i]
                model, loss, acc = self.normal_train(
                    model, dataloader)
                # client_model.append(model)
                self.args_train['client_loss'][i].extend(loss)
                self.args_train['client_accuracy'][i].extend(acc)
        elif self.trainer == 2:
            # 仅在本地数据和客户端上训练，不进行蒸馏
            # client_model = []
            # loss_ = list_same_term(self.args_train['num_server_client'])
            # acc_ = deepcopy(loss_)
            for i in self.args_train['client_idx']:
                message = f'   -client {i}'
                self.args_train['log'].info(message)
                self.args_train['log'].info(message)
                model = deepcopy(self.args_train['client_model'][i])
                dataloader = self.args_train['train_dataloader'][i]
                model, loss, acc = self.normal_train(model, dataloader)
                self.args_train['client_loss'][i].extend(loss)
                self.args_train['client_accuracy'][i].extend(acc)
                model, loss, acc = self.normal_train(
                    model, self.args_train['public_dataloader'])
                # client_model.append(model)
                self.args_train['client_loss'][i].extend(loss)
                self.args_train['client_accuracy'][i].extend(acc)
        elif self.trainer == 3:
            # 本地训练+加权蒸馏
            # client_model = []
            # loss_ = list_same_term(self.args_train['num_server_client'])
            # acc_ = deepcopy(loss_)
            # neighbor_model = self.args_train['neighbor']
            for i in self.args_train['client_idx']:
                message = f'   -client {i}'
                self.args_train['log'].info(message)
                model = deepcopy(self.args_train['client_model'][i])
                dataloader = self.args_train['train_dataloader'][i]
                model, loss, acc = self.normal_train(
                    model, dataloader)
                self.args_train['client_loss'][i].extend(loss)
                self.args_train['client_accuracy'][i].extend(acc)
                model, loss, acc = self.weighted_distill_train(
                    model, self.args_train['neighbor'], weight)
                # client_model.append(model)
                self.args_train['client_loss'][i].extend(loss)
                self.args_train['client_accuracy'][i].extend(acc)
        elif self.trainer == 4:
            # 本地训练+逐个蒸馏
            # client_model = []
            # loss_ = list_same_term(self.args_train['num_server_client'])
            # acc_ = deepcopy(loss_)
            # neighbor_model = [client_model[i] for i in self.args_train['server_client'][self.args_train['neighbor']]]
            n = len(self.args_train['server_model'])
            weight = [1/n for _ in range(n)]
            for i in self.args_train['client_idx']:
                message = f'   -client {i}'
                self.args_train['log'].info(message)
                model = deepcopy(self.args_train['client_model'][i])
                dataloader = self.args_train['train_dataloader'][i]
                model, loss, acc = self.normal_train(
                    model, dataloader)
                self.args_train['client_loss'][i].extend(loss)
                self.args_train['client_accuracy'][i].extend(acc)
                model, loss, acc = self.single_distill_train(
                    model, self.args_train['neighbor'])
                self.args_train['client_loss'][i].extend(loss)
                self.args_train['client_accuracy'][i].extend(acc)
        else:
            raise ValueError('trainer error.')
        # return client_model

    def normal_train(self, model, dataloader):
        loss_ = []
        acc_ = []
        for epoch in range(self.args.num_client_train):
            model, loss = train_model(
                model=model,
                dataloader=dataloader,
                device=self.args_train['device'])
            acc = eval_model(
                model=model,
                dataloader=self.args_train['test_dataloader'],
                device=self.args_train['device']).item()
            acc_.append(acc)
            message = '    |{:^15}: {}, acc {:.3f}'.format(
                'local epoch', epoch, acc)
            self.args_train['log'].info(message)
            loss_.extend(loss)
        return model, loss_, acc_

    def weighted_distill_train(self, model, neighbor_model, weight):
        loss_ = []
        acc_ = []
        for epoch in range(self.args.num_public_train):
            model, loss = train_model_disti_weighted(
                model=model,
                neighbor_server_model=neighbor_model,
                weight=weight,
                dataloader=self.args_train['public_dataloader'],
                alpha=self.args.alpha,
                T=self.args.T,
                device=self.args_train['device'],
                num_target=self.args_train['num_target'])
            loss_.extend(loss)
            acc = eval_model(
                model=model,
                dataloader=self.args_train['test_dataloader'],
                device=self.args_train['device']).item()
            acc_.append(acc)
            message = '    |{:^15}: {}, acc {:.3f}'.format(
                'distill epoch', epoch, acc)
            self.args_train['log'].info(message)
        return model, loss_, acc_

    def single_distill_train(self, model, neighbor_model):
        loss_ = []
        acc_ = []
        for epoch in range(self.args.num_public_train):
            for model_ in neighbor_model:
                model, loss, acc = train_model_disti_single(
                    model=model,
                    teacher_model=model_,
                    dataloader=self.args_train['public_dataloader'],
                    alpha=self.args.alpha,
                    T=self.args.T,
                    device=self.args_train['device'])
                loss_.append(loss)
                acc = eval_model(
                    model=model,
                    dataloader=self.args_train['test_dataloader'],
                    device=self.args_train['device']).item()
            message = '    |{:^15}: {}, acc {:.3f}'.format(
                'distill epoch', epoch, acc)
            self.args_train['log'].info(message)
            acc_.append(acc)
        return model, loss_, acc_
