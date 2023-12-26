# %% intial
import torch
import numpy
import time
# from mpi4py import MPI


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

args = get_args()
args.device = device

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# alpha = 0.5
# T = 2
# num_server_commu = 15
# num_client_commu = 10
# num_client_train = 10
# num_public_train = 10
# batch_size = 200
# dataset = 'mnist'

# num_all_client = 9
# num_all_server = 3
# num_client_data = 1000
# num_public_data = 50
# proportion = 0.8
# server_client = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
# neighbor_server = [[1], [2], [0]]
all_client = [i for i in range(args.num_all_client)]
all_server = [i for i in range(args.num_all_server)]
num_server_client = args.num_all_client // args.num_all_server

message = '\n{}{:^19}:{:^7}\n\
    {:^19}:{:^7}\n\
    {:^19}:{:^7}\n\
    {:^19}:{:^7}\n\
    {:^19}:{:^7}\n\
    {:^19}:{:^7}\n\
    {:^19}:{:^7}\n\
    {:^19}:{:^7}\n\
    {:^19}:{:^7}\n\
    {:^19}:{:^7}\n\
    {:^19}:{:^7}\n\
    {:^19}:{:^7}\n\
    {:^19}:{:^7}\n\
    {:^19}:{:^7}\n'.format(
    ' '*4,
    'alpha', args.alpha,
    'T', args.T,
    'num_server_commu', args.num_server_commu,
    'num_client_commu', args.num_client_commu,
    'num_client_train', args.num_client_train,
    'num_public_train', args.num_public_train,
    'batch_size', args.batch_size,
    'dataset', args.dataset,
    'model_select', args.model_select,
    'num_all_client', args.num_all_client,
    'num_all_server', args.num_all_server,
    'num_client_data', args.num_client_data,
    'num_public_data', args.num_public_data,
    'proportion', args.proportion)
log.info(message)
# %% 原始数据处理
train_dataset_o, test_dataset_o, c, h, w = get_dataset(args.dataset)
TrainDatasetSplited = SplitData(train_dataset_o, args)
all_target = TrainDatasetSplited.targets
num_target = TrainDatasetSplited.num_target

client_main_target = numpy.random.choice(
    all_target, args.num_all_client, replace=False).tolist()
train_dataset_client = TrainDatasetSplited.server_non_iid(
    num_server=args.num_all_server,
    num_server_client=num_server_client,
    num_client_data=args.num_client_data, client_main_target=client_main_target,
    proportion=args.proportion)
train_dataloader = list_same_term(args.num_all_client)
for i, dataset_ in enumerate(train_dataset_client):
    train_dataloader[i] = DataLoader(
        dataset=dataset_,
        batch_size=args.batch_size,
        shuffle=True)
[public_dataset, test_dataset] = split_parts_random(
    test_dataset_o, [args.num_public_data, int(len(test_dataset_o)) - args.num_public_data])
public_dataloader = DataLoader(
    dataset=public_dataset,
    batch_size=args.batch_size,
    shuffle=True)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=300,
    shuffle=True)

# %% 模型初始化
if args.model_select == 1:
    model = CNN(h, w, c, num_target)
elif args.model_select == 2:
    model = LeNet5(h, w, c, num_target)
else:
    raise ValueError('model error.')

client_model = list_same_term(args.num_all_client, model)
server_model = deepcopy(client_model)
server_accuracy = list_same_term(args.num_all_server)
client_accuracy = list_same_term(args.num_all_client)
server_client_model = deepcopy(server_accuracy)
# server_model_distillation_accuracy = deepcopy(server_accuracy)
client_loss = deepcopy(client_accuracy)

# %% 模型训练

keys = ['server_model', 'train_dataloader', 'test_dataloader',
        'public_dataloader', 'log', 'client_model', 'num_target', 'neighbor', 'client_idx', 'client_accuracy', 'client_loss']
values = [server_model, train_dataloader, test_dataloader,
          public_dataloader, log, None, num_target, None, None, client_accuracy, client_loss]
args_train = dict(zip(keys, values))
# %%
# 对每个服务器通讯幕进行循环
algorithm = args.algorithm
# algorithm = rank
for epoch_server_commu in range(args.num_server_commu):
    log.info('-'*50)
    log.info('|epoch_server_commu: {}/{}'.format(epoch_server_commu,
             args.num_server_commu))
    # 所有边缘服务器分别协调其客户端进行多轮联邦学习
    for epoch_client_commu in range(args.num_client_commu):
        message = ' |epoch_client_commu: {}/{}'.format(
            epoch_client_commu, args.num_client_commu)
        log.info(message)
        # 所有边缘服务器分别协调其客户端进行联邦学习
        neighbor_model = []
        for server in all_server:
            # 每个服务器下单客户端分别训练
            message = f'  |server: {server}'
            log.info(message)
            args_train['client_idx'] = args.server_client[server]
            args_train['client_model'] = client_model
            if algorithm == 0:
                if epoch_server_commu == 0:
                    client_model = ServerTrain(args, args_train, 1).train
                else:
                    for i in args.neighbor_server[server]:
                        neighbor_model.append(server_model[i])
                    args_train['neighbor'] = neighbor_model
                    client_model = ServerTrain(args, args_train, 3).train
            if algorithm == 1:
                if epoch_server_commu == 0:
                    client_model = ServerTrain(args, args_train, 1).train
                else:
                    for i in args.neighbor_server[server]:
                        neighbor_model.append(server_model[i])
                    args_train['neighbor'] = neighbor_model
                    client_model = ServerTrain(args, args_train, 4).train
            if algorithm == 2 or algorithm == 3:
                client_model = ServerTrain(args, args_train, 1).train
            if args.algorithm == 4:
                client_model = ServerTrain(args, args_train, 2).train
            # 在单个服务器下客户端训练完成后更新该服务器下客户端的模型
            server_client_model[server] = [
                client_model[client] for client in args.server_client[server]]
            # 聚合获得单个服务器模型并下发
            weight_server = [1/3, 1/3, 1/3]
            server_model[server] = aggregate(
                server_client_model[server], weight_server)
            for client in args.server_client[server]:
                client_model[client] = deepcopy(server_model[server])
            # 评估单个服务器模型
            acc_server = eval_model(
                model=server_model[server],
                dataloader=test_dataloader,
                device=device)
            message = '|servers comunicated: {}, server aggregated: {}, acc_server {}: {:.3f}.'.format(
                epoch_server_commu, epoch_client_commu, server, acc_server)
            log.info(message)
            log.info('-'*50)
            server_accuracy[server].append(acc_server)
        if args.algorithm == 2:
            weight_list = [[1/3, 1/3, 1/3], [1/3, 1/3, 1/3], [1/3, 1/3, 1/3]]
            server_model = server_communicate(server_model, weight_list)
            for server in all_server:
                for client in args.server_client[server]:
                    client_model[client] = deepcopy(server_model[server])
    message = '{:^50}'.format('********  servers comunicates  ********')
    log.info(message)

# %% 保存
save_data = {'server_model': server_model,
             'server_acc': server_accuracy,
             'client_model': client_model,
             'client_acc': client_accuracy,
             'client_loss': client_loss}
# save_file(args, save_data, rank)
save_file(args, save_data)
