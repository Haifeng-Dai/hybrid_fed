# %% intial
import torch
import numpy
import time
import sys
import os
# from mpi4py import MPI

from torch.utils.data import DataLoader

from utils.model_util import *
from utils.data_util import *
from utils.lib_util import *
from utils.train_util import *

t = time.localtime()
log_path = f'./tmp.log'
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

server_client = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
neighbor_server = [[1], [2], [0]]
all_client = [i for i in range(args.num_all_client)]
all_server = [i for i in range(args.num_all_server)]
num_server_client = args.num_all_client // args.num_all_server

message = '\n\
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
    {:^19}:{:^7}\n\
    {:^19}:{:^7}\n\
    {:^19}:{:^7}\n'.format(
    'alpha', args.alpha,
    'T', args.T,
    'algorithm', args.algorithm,
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

# if os.path.exists(log_path):
#     message = 'file exists.'
#     log.info(message)
#     sys.exit()

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
# keys = ['server_model',
#         'train_dataloader',
#         'test_dataloader',
#         'train_test_dataloader',
#         'public_dataloader',
#         'log',
#         'client_model',
#         'num_target',
#         'neighbor',
#         'client_idx',
#         'client_accuracy',
#         'client_loss',
#         'train_accuracy']
# values = [server_model,
#           train_dataloader,
#           test_dataloader,
#           train_test_dataloader,
#           public_dataloader,
#           log,
#           None,
#           num_target,
#           None,
#           None,
#           client_accuracy,
#           client_loss,
#           train_accuracy]
# args_train = dict(zip(keys, values))

client_model_save = dict.fromkeys([i for i in range(args.num_client_commu)])
server_model_save = dict.fromkeys([i for i in range(args.num_server_commu)])
weight_server = list_same_term(args.num_all_server, 1/args.num_all_server)
weight_list = list_same_term(args.num_all_server, weight_server)

save_data = {'args': args,
             'num_target': num_target,
             'num_server_client': num_server_client,
             'train_dataloader': train_dataloader,
             'test_dataloader': test_dataloader,
             'train_test_dataloader': train_test_dataloader,
             'public_dataloader': public_dataloader,
             'server_client': server_client,
             'neighbor_server': neighbor_server,
             'all_server': all_server,
             'client_model': client_model,
             'server_model': server_model,
             'client_loss': client_loss,
             'client_accuracy': client_accuracy,
             'server_accuracy': server_accuracy,
             'all_client': all_client,
             'log_path': log_path}
message = 'save data'
file_path = './test.pt'
torch.save(save_data, file_path)
log.info(message)
