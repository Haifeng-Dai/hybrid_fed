# %% intial
import torch
import numpy
import time

from torch.utils.data import DataLoader

from utils.model_util import *
from utils.data_util import *
from utils.lib_util import *
from utils.train_util import *
from trainer import *

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

# %% 参数定义
args = get_args()
setup_seed(args.seed)
args.device = get_device(log)

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
{'proportion':^17}:{args.proportion:^7}\n\
{'num_workers':^17}:{args.num_workers:^7}"
log.info(message)

# %% 原始数据处理
train_dataset_o, test_dataset_o, c, h, w = get_dataset(args.dataset)
target_list = {0: [0, 1, 2], 1: [3, 4, 5], 2: [6, 7, 8, 9]}
num_target, dataset_train, validate_dataloader = split_dataset(
    train_dataset_o, target_list, args)
[public_dataset, test_dataset] = split_parts_random(
    test_dataset_o, [args.num_public_data, int(len(test_dataset_o)) - args.num_public_data])
public_dataloader = DataLoader(
    dataset=public_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    pin_memory=True,
    num_workers=args.num_workers)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=args.batch_size,
    pin_memory=True,
    num_workers=args.num_workers)
train_dataloader = {}
for client in all_client:
    train_dataloader[client] = DataLoader(
        dataset=dataset_train,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers)
regular_dataset = deepcopy(dataset_train)
regular_dataloader = {}
for client in all_client:
    regular_dataset[client].extend(public_dataset)
    regular_dataloader[client] = DataLoader(
        dataset=regular_dataset[client],
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=args.num_workers)

# %% 模型初始化
client_model, server_model = intial_model(
    args, num_target, num_server_client, c, h, w)
client_accuracy = list_same_term(args.num_all_client)
validate_accuracy = list_same_term(args.num_all_client)
client_loss = deepcopy(client_accuracy)
weight_server = list_same_term(args.num_all_server, 1/args.num_all_server)
weight_list = list_same_term(args.num_all_server, weight_server)

# %% 模型训练
keys = ['server_model',
        'train_dataloader',
        'test_dataloader',
        'validate_dataloader',
        'public_dataloader',
        'log',
        'num_target',
        'client_accuracy',
        'client_loss',
        'validate_accuracy',
        'weight_list',
        'weight_server',
        'server_client',
        'all_server',
        'client_model',
        'target_list',
        'public_dataset',
        'regular_dataloader']
values = [server_model,
          train_dataloader,
          test_dataloader,
          validate_dataloader,
          public_dataloader,
          log,
          num_target,
          client_accuracy,
          client_loss,
          validate_accuracy,
          weight_list,
          weight_server,
          server_client,
          all_server,
          client_model,
          target_list,
          public_dataset,
          regular_dataloader]
args_train = dict(zip(keys, values))
server_accuracy = Trainer(neighbor_server, args, args_train).train

# %% 保存
save_data = {'args': args,
             'server_acc': server_accuracy,
             'client_acc': args_train['client_accuracy'],
             'validate_acc': args_train['validate_accuracy'],
             'client_loss': args_train['client_loss']}
save_file(args, save_data, log)
