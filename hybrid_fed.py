# %% intial
import torch
import numpy
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from utils.model_util import LeNet5, CNN
from utils.data_util import *
from utils.lib_util import *
from utils.train_util import *

torch.set_printoptions(
    precision=2,
    threshold=1000,
    edgeitems=5,
    linewidth=1000,
    sci_mode=False)
# 是否使用显卡加速
if torch.cuda.is_available():
    device = 'cuda'
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True
        print('cudnn', end=' and ')
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'
print(device)

# %% 参数定义
alpha = 0.5
T = 2
num_server_commu = 100
num_client_commu = 10
num_client_train = 10
num_public_train = 10
batch_size = 100

num_all_client = 9
num_all_server = 3
num_client_data = 1000
all_client = number_list(num_all_client)
all_server = number_list(num_all_server)
server_client = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
neighbor_server = [[1], [2], [0]]

# %% 原始数据处理
save_path = './data/dealed-data/'
dataset = 'mnist'
train_dataset_o, test_dataset_o, c, h, w = get_dataset(dataset)
TrainDatasetSplited = SplitData(train_dataset_o)
all_target = TrainDatasetSplited.targets
num_target = TrainDatasetSplited.num_target

# target_list = [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]
# train_public_dataset_client = TrainDatasetSplited.part_target(
#     num_all_client, num_client_data, target_list)
# train_dataset_client = train_public_dataset_client[:9]
# public_dataset = train_public_dataset_client[-1][:200]

proportion = 0.8
client_main_target = numpy.random.choice(
    all_target, num_all_client, replace=False).tolist()
train_dataset_client = TrainDatasetSplited.server_non_iid(
    num_server=num_all_server,
    num_server_client=num_all_client // num_all_server,
    num_client_data=num_client_data, client_main_target=client_main_target, proportion=proportion)

[public_dataset, test_dataset] = split_parts_random(
    test_dataset_o, [50, 9950])
public_dataloader = DataLoader(
    dataset=public_dataset,
    batch_size=batch_size,
    shuffle=True)
test_dataloader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=True)

# %% 模型初始化
initial_model = CNN(h, w, c, num_target)

client_model = list_same_term(num_all_client, initial_model)
server_model = list_same_term(num_all_server, initial_model)
server_accuracy = list_same_term(num_all_server)
client_accuracy = list_same_term(num_all_client)
server_client_model = list_same_term(num_all_server)
server_model_distillation_accuracy = list_same_term(num_all_server)

# %% 模型训练
# 对每个服务器通讯幕进行循环
print('-'*50)
for epoch_server_commu in range(num_server_commu):
    print('|epoch_server_commu', epoch_server_commu, '/', num_server_commu)
    # 所有边缘服务器分别协调其客户端进行多轮联邦学习
    for epoch_client_commu in range(num_client_commu):
        print(' -|epoch_client_commu', epoch_client_commu, '/', num_client_commu)
        # 所有边缘服务器分别协调其客户端进行联邦学习
        for server in all_server:
            # 每个服务器下单客户端分别训练
            for client in server_client[server]:
                print('  -|server', server, '/', num_all_server,
                      'client', client, '/', len(server_client[server]))
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
                    print('    |local epoch,', epoch, 'acc', acc)
                # 单个服务器下的客户端在公开数据集上进行num_public_train轮训练
                if epoch_server_commu != 0:
                    neighbor_server_model = [
                        server_model_distillation[server] for server in neighbor_server[server]]
                    weight = torch.tensor(
                        [1/len(neighbor_server_model) for _ in neighbor_server_model])
                    for epoch in range(num_public_train):
                        # client_model[client] = train_model_disti_weighted(
                        #     model=client_model[client],
                        #     neighbor_server_model=neighbor_server_model,
                        #     weight=weight,
                        #     dataloader=public_dataloader,
                        #     alpha=alpha,
                        #     T=2,
                        #     device=device,
                        #     num_target=num_target)
                        # acc = eval_model(
                        #     model=client_model[client],
                        #     dataloader=test_dataloader,
                        #     device=device).item()
                        # print('     distill epoch,', epoch, 'acc', acc)
                        for model_ in neighbor_server_model:
                            client_model[client] = train_model_disti_single(
                                model=client_model[client],
                                teacher_model=model_,
                                dataloader=public_dataloader,
                                alpha=alpha,
                                T=T,
                                device=device)
                            acc = eval_model(
                                model=client_model[client],
                                dataloader=test_dataloader,
                                device=device).item()
                        print('    |distill epoch,',
                              epoch, 'acc', acc)
                # 在训练后评估该服务器下的客户端
                client_accuracy[client].append(eval_model(
                    model=client_model[client],
                    dataloader=test_dataloader,
                    device=device))
            print('-'*50)
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
            print(
                f'|servers comunicated {epoch_server_commu}, server aggregated {epoch_client_commu}, server {server} acc_server: {acc_server:.3f}')
            print('-'*50)
            server_accuracy[server].append(acc_server)
    # 服务器在多轮更新联邦学习后固定用于蒸馏的模型
    # print('-'*50)
    print('|servers comunicates.')
    server_model_distillation = deepcopy(server_model)
    # # 评估该蒸馏模型
    # for server in all_server:
    #     acc_server_distill = eval_model(
    #         model=server_model_distillation[server],
    #         dataloader=test_dataloader,
    #         device=device)
    #     print('distill server', server, 'acc_server_distill', acc_server_distill)
    #     server_model_distillation_accuracy[server].append(acc_server_distill)
    print('-'*100)

# %% 作图
x = [i for i in range(num_server_commu * num_client_commu)]
line_list = []
for server in all_server:
    line_list.append(plt.plot(x, server_accuracy[server]))

plt.show()
