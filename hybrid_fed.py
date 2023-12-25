# %% intial
import torch
import numpy
import time
import argparse
import matplotlib.pyplot as plt

from trainer import *

from torch.utils.data import DataLoader

from utils.model_util import LeNet5, CNN
from utils.data_util import *
from utils.lib_util import *
from utils.train_util import *

t = time.localtime()
log_path = f'./log/{t.tm_year}-{t.tm_mon}-{t.tm_mday}-{t.tm_hour}-{t.tm_min}.log'
log = get_logger(log_path)

torch.set_printoptions(
    precision=2,
    threshold=1000,
    edgeitems=5,
    linewidth=1000,
    sci_mode=False)
# 是否使用显卡加速
if torch.cuda.is_available():
    device = 'cuda'
    log.info(f'device {device} is used.')
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        log.info('cudnn is actived.')
elif torch.backends.mps.is_available():
    device = 'mps'
    log.info(f'device {device} is used.')
else:
    device = 'cpu'
    log.info(f'device {device} is used.')


# %% 参数定义

parser = argparse.ArgumentParser(description='save results.')
parser.add_argument('--dataset', type=str, default='mnist',
                    help='the used dataset.')
parser.add_argument('--alpha', type=int, default=0.5,
                    help='trade-off parameters of distillation.')
parser.add_argument('--T', type=int, default=2,
                    help='temperature of distillation.')
parser.add_argument('--num_all_client', type=int, help='num of all client.')
parser.add_argument('--num_all_server', type=int, help='num of all server.')
parser.add_argument('--distil_way', type=str,
                    default='weighted', help='the way of distillation.')
parser.add_argument('--batch_size', type=int, default=100,
                    help='batch size of trainning.')
parser.add_argument('--num_client_data', type=int,
                    default=1000, help='number of client datas.')
parser.add_argument('--num_server_commu', type=int, default=10,
                    help='number of server communications.')
parser.add_argument('--num_client_commu', type=int, default=10,
                    help='number of clients communicate with servers.')
parser.add_argument('--num_client_train', type=int, default=10,
                    help='number of client train in local data.')
parser.add_argument('--num_public_train', type=int, default=1,
                    help='number of client distillation in public data.')
parser.add_argument('--model_select', type=int, default=1,
                    help='select the model group.')

args = parser.parse_args()

args.device = device
print(args.device)
alpha = 0.5
T = 2
num_server_commu = 1
num_client_commu = 1
num_client_train = 1
num_public_train = 1
batch_size = 100
dataset = 'mnist'
model = 'cnn'

num_all_client = 9
num_all_server = 3
num_client_data = 1000
num_public_data = 50
proportion = 0.8
server_client = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
neighbor_server = [[1], [2], [0]]
all_client = [i for i in range(num_all_client)]
all_server = [i for i in range(num_all_server)]
num_server_client=num_all_client // num_all_server

tab = ' ' * 25
message = '{:^19}:{:^7}\n\
    {}{:^19}:{:^7}\n\
    {}{:^19}:{:^7}\n\
    {}{:^19}:{:^7}\n\
    {}{:^19}:{:^7}\n\
    {}{:^19}:{:^7}\n\
    {}{:^19}:{:^7}\n\
    {}{:^19}:{:^7}\n\
    {}{:^19}:{:^7}\n\
    {}{:^19}:{:^7}\n\
    {}{:^19}:{:^7}\n\
    {}{:^19}:{:^7}\n\
    {}{:^19}:{:^7}\n\
    {}{:^19}:{:^7}\n'.format(
    'alpha', alpha, ' '*25,
    'T', T, ' '*25,
    'num_server_commu', num_server_commu, tab,
    'num_client_commu', num_client_commu, tab,
    'num_client_train', num_client_train, tab,
    'num_public_train', num_public_train, tab,
    'batch_size', batch_size, tab,
    'dataset', dataset, tab,
    'model', model, tab,
    'num_all_client', num_all_client, tab,
    'num_all_server', num_all_server, tab,
    'num_client_data', num_client_data, tab,
    'num_public_data', num_public_data, tab,
    'proportion', proportion, tab)
log.info(message)
# %% 原始数据处理
train_dataset_o, test_dataset_o, c, h, w = get_dataset(dataset)
TrainDatasetSplited = SplitData(train_dataset_o, args)
all_target = TrainDatasetSplited.targets
num_target = TrainDatasetSplited.num_target

# target_list = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
# train_public_dataset_client = TrainDatasetSplited.part_target(
#     num_all_client, num_client_data, target_list)
# train_dataset_client = train_public_dataset_client[:9]
# public_dataset = train_public_dataset_client[-1][:200]

client_main_target = numpy.random.choice(
    all_target, num_all_client, replace=False).tolist()
train_dataset_client = TrainDatasetSplited.server_non_iid(
    num_server=num_all_server,
    num_server_client=num_server_client,
    num_client_data=num_client_data, client_main_target=client_main_target, proportion=proportion)
train_dataloader = list_same_term(num_all_client)
for i, dataset_ in enumerate(train_dataset_client):
    train_dataloader[i] = DataLoader(
        dataset=dataset_,
        batch_size=args.batch_size,
        shuffle=True)
[public_dataset, test_dataset] = split_parts_random(
    test_dataset_o, [num_public_data, int(1e4)-num_public_data])
public_dataloader = DataLoader(
    dataset=public_dataset,
    batch_size=batch_size,
    shuffle=True)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True)

# %% 模型初始化
if args.model_select == 1:
    model = CNN(h, w, c, num_target)
elif args.model_select == 2:
    model = LeNet5(h, w, c, num_target)
else:
    raise ValueError('model error.')

client_model = list_same_term(num_all_client, model)
server_model = deepcopy(client_model)
server_accuracy = list_same_term(num_all_server)
client_accuracy = list_same_term(num_all_client)
server_client_model = deepcopy(server_accuracy)
server_model_distillation_accuracy = deepcopy(server_accuracy)
client_loss = deepcopy(client_accuracy)

# for server in range(num_all_server):
#     server_client_model[server] = [
#         client_model[client] for client in server_client[server]]
# %% 模型训练

keys = ['server_model', 'train_dataloader', 'test_dataloader',
        'public_dataloader', 'device', 'log', 'client_model', 'num_target', 'server_client', 'neighbor', 'client_idx', 'num_server_client', 'client_accuracy', 'client_loss']
values = [server_model, train_dataloader, test_dataloader,
          public_dataloader, device, log, client_model, num_target, server_client, None, None, num_server_client, client_accuracy, client_loss]
args_train = dict(zip(keys, values))
# %%
# 对每个服务器通讯幕进行循环
for epoch_server_commu in range(num_server_commu):
    log.info('-'*50)
    log.info('|epoch_server_commu: {}/{}'.format(epoch_server_commu, num_server_commu))
    # 所有边缘服务器分别协调其客户端进行多轮联邦学习
    for epoch_client_commu in range(num_client_commu):
        message = ' |epoch_client_commu: {}/{}'.format(
            epoch_client_commu, num_client_commu)
        log.info(message)
        # 所有边缘服务器分别协调其客户端进行联邦学习
        neighbor_model = []
        for server in all_server:
            # 每个服务器下单客户端分别训练
            message = f'  |server: {server}'
            log.info(message)
            args_train['client_idx'] = server_client[server]
            if epoch_server_commu == 0:
                # client_model = Server(args, args_train, 1).train()
                Server(args, args_train, 1).train()
            else:
                for i in neighbor_server[server]:
                    neighbor_model.append(server_model[i])
                args_train['neighbor'] = neighbor_model
                # client_model = Server(args, args_train, 3).train()
                Server(args, args_train, 3).train()
            # for client in server_client[server]:
            #     message = '  -|server: {}/{}'.format(server, num_all_server)
            #     log.info(message)
            #     # 单个服务器下的客户端在私有数据集上进行num_client_train轮训练
            #     client_model[client] = deepcopy(server_model[server])
            #     for epoch in range(num_client_train):
            #         client_model[client], loss = train_model(
            #             model=client_model[client],
            #             dataloader=train_dataloader[client],
            #             device=device)
            #         acc = eval_model(
            #             model=client_model[client],
            #             dataloader=test_dataloader,
            #             device=device).item()
            #         message = '    |{:^15}: {}, acc {:.3f}'.format(
            #             'local epoch', epoch, acc)
            #         log.info(message)
            #         print(loss)
            #     # 单个服务器下的客户端在公开数据集上进行num_public_train轮训练
            #     if epoch_server_commu != 0:
            #         neighbor_server_model = [
            #             server_model_distillation[server] for server in neighbor_server[server]]
            #         weight = torch.tensor(
            #             [1/len(neighbor_server_model) for _ in neighbor_server_model])
            #         for epoch in range(num_public_train):
            #             # client_model[client], _ = train_model_disti_weighted(
            #             #     model=client_model[client],
            #             #     neighbor_server_model=neighbor_server_model,
            #             #     weight=weight,
            #             #     dataloader=public_dataloader,
            #             #     alpha=alpha,
            #             #     T=T,
            #             #     device=device,
            #             #     num_target=num_target)
            #             # acc = eval_model(
            #             #     model=client_model[client],
            #             #     dataloader=test_dataloader,
            #             #     device=device).item()
            #             # message = '    |{:^15}: {}, acc {:.3f}'.format('distill epoch', epoch, acc)
            #             # log.info(message)
            #             for model_ in neighbor_server_model:
            #                 client_model[client], loss = train_model_disti_single(
            #                     model=client_model[client],
            #                     teacher_model=model_,
            #                     dataloader=public_dataloader,
            #                     alpha=alpha,
            #                     T=T,
            #                     device=device)
            #                 acc = eval_model(
            #                     model=client_model[client],
            #                     dataloader=test_dataloader,
            #                     device=device).item()
            #             message = '    |{:^15}: {}, acc {:.3f}'.format(
            #                 'distill epoch', epoch, acc)
            #             log.info(message)
            #             print(loss)
            # client_accuracy[client].append(eval_model(
            #     model=client_model[client],
            #     dataloader=test_dataloader,
            #     device=device))
            # log.info('-'*50)
            # 在单个服务器下客户端训练完成后更新该服务器下客户端的模型
            server_client_model[server] = [
                client_model[client] for client in server_client[server]]
            # 聚合获得单个服务器模型并下发
            server_model[server] = EdgeServer(
                server_client_model[server]).average()
            for client in server_client[server]:
                client_model[client] = deepcopy(server_model)
            # 评估单个服务器模型
            acc_server = eval_model(
                model=server_model[server],
                dataloader=test_dataloader,
                device=device).item()
            message = '|servers comunicated: {}, server aggregated: {}, acc_server {}: {:.3f}.'.format(
                epoch_server_commu, epoch_client_commu, server, acc_server)
            log.info(message)
            log.info('-'*50)
            server_accuracy[server].append(acc_server)
    # 服务器在多轮更新联邦学习后固定用于蒸馏的模型
    # print('-'*50)
    log.info('******** servers comunicates ********')
    server_model_distillation = deepcopy(server_model)
    # # 评估该蒸馏模型
    # for server in all_server:
    #     acc_server_distill = eval_model(
    #         model=server_model_distillation[server],
    #         dataloader=test_dataloader,
    #         device=device)
    #     print('distill server', server, 'acc_server_distill', acc_server_distill)
    #     server_model_distillation_accuracy[server].append(acc_server_distill)

# %% 保存

# save_file(args, )
