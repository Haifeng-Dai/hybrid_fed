# %% intial
import torch
import numpy
import time
import matplotlib.pyplot as plt

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
alpha = 0.5
T = 2
num_server_commu = 15
num_client_commu = 10
num_client_train = 10
num_public_train = 10
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

tab = ' ' * 25
message = '{:^20}:{:^20}\n\
    {}{:^20}:{:^20}\n\
    {}{:^20}:{:^20}\n\
    {}{:^20}:{:^20}\n\
    {}{:^20}:{:^20}\n\
    {}{:^20}:{:^20}\n\
    {}{:^20}:{:^20}\n\
    {}{:^20}:{:^20}\n\
    {}{:^20}:{:^20}\n\
    {}{:^20}:{:^20}\n\
    {}{:^20}:{:^20}\n\
    {}{:^20}:{:^20}\n\
    {}{:^20}:{:^20}\n\
    {}{:^20}:{:^20}\n'.format(
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
save_path = './data/dealed-data/'
train_dataset_o, test_dataset_o, c, h, w = get_dataset(dataset)
TrainDatasetSplited = SplitData(train_dataset_o)
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
    num_server_client=num_all_client // num_all_server,
    num_client_data=num_client_data, client_main_target=client_main_target, proportion=proportion)

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
if model == 'cnn':
    initial_model = CNN(h, w, c, num_target)
elif model == 'lenet5':
    initial_model = LeNet5(h, w, c, num_target)
else:
    raise ValueError('model error.')

client_model = list_same_term(num_all_client, initial_model)
server_model = list_same_term(num_all_server, initial_model)
server_accuracy = list_same_term(num_all_server)
client_accuracy = list_same_term(num_all_client)
server_client_model = list_same_term(num_all_server)
server_model_distillation_accuracy = list_same_term(num_all_server)

# %% 模型训练
# 对每个服务器通讯幕进行循环
for epoch_server_commu in range(num_server_commu):
    log.info('-'*50)
    log.info('|epoch_server_commu: {}/{}'.format(epoch_server_commu, num_server_commu))
    # 所有边缘服务器分别协调其客户端进行多轮联邦学习
    for epoch_client_commu in range(num_client_commu):
        message = ' -|epoch_client_commu: {}/{}'.format(epoch_client_commu, num_client_commu)
        log.info(message)
        # 所有边缘服务器分别协调其客户端进行联邦学习
        for server in all_server:
            # 每个服务器下单客户端分别训练
            for client in server_client[server]:
                message ='  -|server: {}/{}, client: {}/{}'.format(server, num_all_server, client, len(server_client[server]))
                log.info(message)
                # 单个服务器下的客户端在私有数据集上进行num_client_train轮训练
                client_model[client] = deepcopy(server_model[server])
                train_dataloader_client = DataLoader(
                    dataset=train_dataset_client[client],
                    batch_size=batch_size,
                    shuffle=True)
                for epoch in range(num_client_train):
                    client_model[client] = train_model(
                        model=client_model[client],
                        dataloader=train_dataloader_client,
                        device=device)
                    acc = eval_model(
                        model=client_model[client],
                        dataloader=test_dataloader,
                        device=device).item()
                    message = '    |{:^15}: {:^2}, acc {:.3f}'.format('local epoch', epoch, acc)
                    log.info(message)
                # 单个服务器下的客户端在公开数据集上进行num_public_train轮训练
                if epoch_server_commu != 0:
                    neighbor_server_model = [
                        server_model_distillation[server] for server in neighbor_server[server]]
                    weight = torch.tensor(
                        [1/len(neighbor_server_model) for _ in neighbor_server_model])
                    for epoch in range(num_public_train):
                        client_model[client] = train_model_disti_weighted(
                            model=client_model[client],
                            neighbor_server_model=neighbor_server_model,
                            weight=weight,
                            dataloader=public_dataloader,
                            alpha=alpha,
                            T=2,
                            device=device,
                            num_target=num_target)
                        acc = eval_model(
                            model=client_model[client],
                            dataloader=test_dataloader,
                            device=device).item()
                        message = '    |{:^15}: {}, acc {:.3f}'.format('distill epoch', epoch, acc)
                        log.info(message)
                        # for model_ in neighbor_server_model:
                        #     client_model[client] = train_model_disti_single(
                        #         model=client_model[client],
                        #         teacher_model=model_,
                        #         dataloader=public_dataloader,
                        #         alpha=alpha,
                        #         T=T,
                        #         device=device)
                        #     acc = eval_model(
                        #         model=client_model[client],
                        #         dataloader=test_dataloader,
                        #         device=device).item()
                        # message = '    |{:^15}: {}, acc {:.3f}'.format('distill epoch', epoch, acc)
                        # log.info(message)
                # 在训练后评估该服务器下的客户端
                client_accuracy[client].append(eval_model(
                    model=client_model[client],
                    dataloader=test_dataloader,
                    device=device))
            log.info('-'*50)
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
            message = '|servers comunicated: {}, server aggregated: {}, server: {} acc_server: {:.3f}.'.format(epoch_server_commu, epoch_client_commu, server, acc_server)
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

# %% 作图
x = [i for i in range(num_server_commu * num_client_commu)]
line_list = []
for server in all_server:
    line_list.append(plt.plot(x, server_accuracy[server]))

plt.show()
