# %% intial
import torch
import numpy
import time

from torch.utils.data import DataLoader

from utils.model_util import *
from utils.data_util import *
from utils.lib_util import *
from utils.train_util import *

t = time.localtime()
log_path = f'./log/{t.tm_year}-{t.tm_mon}-{t.tm_mday}/'
if not os.path.exists(log_path):
    os.makedirs(log_path)
log_path += f'{t.tm_hour}-{t.tm_min}-{t.tm_sec}.log'
log = get_logger(log_path)

torch.set_printoptions(
    precision=2,
    threshold=1000,
    edgeitems=5,
    linewidth=1000,
    sci_mode=False)

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

server_client = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
neighbor_server = [[1], [2], [0]]
all_client = [i for i in range(args.num_all_client)]
all_server = [i for i in range(args.num_all_server)]
num_server_client = args.num_all_client // args.num_all_server

message = f"\n\
{'alpha':^17}:{args.alpha:^7}\n\
{'T':^17}:{args.T:^7}\n\
{'algorithm':^17}:{args.algorithm:^7}\n\
{'num_server_commu':^17}:{args.num_server_commu:^7}\n\
{'num_client_commu':^17}:{args.num_client_commu:^7}\n\
{'num_client_train':^17}:{args.num_client_train:^7}\n\
{'num_public_train':^17}:{args.num_public_train:^7}\n\
{'batch_size':^17}:{args.batch_size:^7}\n\
{'dataset':^17}:{args.dataset:^7}\n\
{'model_select':^17}:{args.model_select:^7}\n\
{'num_all_client':^17}:{args.num_all_client:^7}\n\
{'num_all_server':^17}:{args.num_all_server:^7}\n\
{'num_client_data':^17}:{args.num_client_data:^7}\n\
{'num_public_data':^17}:{args.num_public_data:^7}\n\
{'proportion':^17}:{args.proportion:^7}"
log.info(message)

# %% 原始数据处理
train_dataset_o, test_dataset_o, c, h, w = get_dataset(args.dataset)
TrainDatasetSplited = SplitData(train_dataset_o)
all_target = TrainDatasetSplited.targets
num_target = TrainDatasetSplited.num_target

client_main_target = numpy.random.choice(
    all_target, args.num_all_client, replace=False).tolist()
train_dataset_client = TrainDatasetSplited.server_non_iid(
    num_server=args.num_all_server,
    num_server_client=num_server_client,
    num_client_data=args.num_client_data,
    client_main_target=client_main_target,
    proportion=args.proportion)
train_dataloader = list_same_term(args.num_all_client)
train_test_dataloader = list_same_term(args.num_all_client)
for i, dataset_ in enumerate(train_dataset_client):
    [dataset_train, dataset_test] = split_parts_random(
        dataset_, [1000, 200])
    train_dataloader[i] = DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4)
    train_test_dataloader[i] = DataLoader(
        dataset=dataset_test,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=4)
[public_dataset, test_dataset] = split_parts_random(
    test_dataset_o, [args.num_public_data, int(len(test_dataset_o)) - args.num_public_data])
public_dataloader = DataLoader(
    dataset=public_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=4)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    pin_memory=True,
    num_workers=4)

# %% 模型初始化
if args.model_select == 1:
    model = CNN(h, w, c, num_target)
    client_model = list_same_term(args.num_all_client, model)
    server_model = list_same_term(args.num_all_server, model)
elif args.model_select == 2:
    model = LeNet5(h, w, c, num_target)
    client_model = list_same_term(args.num_all_client, model)
    server_model = list_same_term(args.num_all_server, model)
elif args.model_select == 3:
    model1 = CNN(h, w, c, num_target)
    model2 = LeNet5(h, w, c, num_target)
    model3 = MLP(h, w, c, 50, num_target)
    server_model = [model1, model2, model3]
    client_model1 = list_same_term(num_server_client, model1)
    client_model2 = list_same_term(num_server_client, model2)
    client_model3 = list_same_term(num_server_client, model3)
    client_model = [client_model1, client_model2, client_model3]
else:
    raise ValueError('model error.')

server_accuracy = list_same_term(args.num_all_server)
client_accuracy = list_same_term(args.num_all_client)
train_accuracy = list_same_term(args.num_all_client)
server_client_model = deepcopy(server_accuracy)
client_loss = deepcopy(client_accuracy)

# %% 模型训练
keys = ['server_model',
        'train_dataloader',
        'test_dataloader',
        'train_test_dataloader',
        'public_dataloader',
        'log',
        'client_model',
        'num_target',
        'neighbor',
        'client_idx',
        'client_accuracy',
        'client_loss',
        'train_accuracy',
        'LR']
values = [server_model,
          train_dataloader,
          test_dataloader,
          train_test_dataloader,
          public_dataloader,
          log,
          None,
          num_target,
          None,
          None,
          client_accuracy,
          client_loss,
          train_accuracy,
          None]
args_train = dict(zip(keys, values))

weight_server = list_same_term(args.num_all_server, 1/args.num_all_server)
weight_list = list_same_term(args.num_all_server, weight_server)

# %% 对每个服务器通讯幕进行循环
d = 2
for epoch_server_commu in range(args.num_server_commu):
    log.info('-'*50)
    log.info('|epoch_server_commu: {}/{}'.format(epoch_server_commu,
             args.num_server_commu))
    args_train['LR'] = 1e-3 / (1 + d * epoch_server_commu)
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
            args_train['client_idx'] = server_client[server]
            args_train['client_model'] = client_model
            if args.algorithm == 0:
                if epoch_server_commu == 0:  # weighted distill
                    client_model = ServerTrain(args, args_train, 1).train
                else:
                    for i in neighbor_server[server]:
                        neighbor_model.append(server_model[i])
                    args_train['neighbor'] = neighbor_model
                    client_model = ServerTrain(args, args_train, 3).train
            if args.algorithm == 1:  # single distill
                if epoch_server_commu == 0:
                    client_model = ServerTrain(args, args_train, 1).train
                else:
                    for i in neighbor_server[server]:
                        neighbor_model.append(server_model[i])
                    args_train['neighbor'] = neighbor_model
                    client_model = ServerTrain(args, args_train, 4).train
            if args.algorithm == 2 or args.algorithm == 3:  # 3仅在训练集上训练，2交换参数
                client_model = ServerTrain(args, args_train, 1).train
            if args.algorithm == 4:  # 不交换参数，在训练集和公开数据集上训练
                client_model = ServerTrain(args, args_train, 2).train
            # 在单个服务器下客户端训练完成后更新该服务器下客户端的模型
            server_model_ = [
                client_model[client] for client in server_client[server]]
            # 聚合获得单个服务器模型并下发
            server_model[server] = aggregate(
                server_model_, weight_server)
            for client in server_client[server]:
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
        if args.algorithm == 2:  # 参数平均
            server_model = server_communicate(server_model, weight_list)
            for server in all_server:
                for client in server_client[server]:
                    client_model[client] = deepcopy(server_model[server])
    message = '{:^50}'.format('********  servers comunicates  ********')
    log.info(message)

# %% 保存
save_data = {'args': args,
             'server_acc': server_accuracy,
             'client_acc': client_accuracy,
             'train_acc': train_accuracy,
             'client_loss': client_loss}
save_file(args, save_data, log)
