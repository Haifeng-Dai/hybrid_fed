import torch
import os
import matplotlib.pyplot as plt

from tqdm import tqdm
from utils.model_util import *
from utils.data_util import *
from utils.lib_util import *
from utils.train_util import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True
torch.set_printoptions(precision=2,
                       threshold=1000,
                       edgeitems=5,
                       linewidth=1000,
                       sci_mode=False)
# 是否使用显卡加速
if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)

# %% 原始数据处理
save_path = './data/dealed-data/'

dataset = 'mnist'
file_path = save_path + dataset + '_train_dataset_splited.pt'
if os.path.exists(file_path):
    [train_dataset_splited, test_dataset_o] = torch.load(file_path)
    print('file existed.')
else:
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    train_dataset_o, test_dataset_o = get_dataset(dataset)
    train_dataset_splited = split_data(train_dataset_o)
    torch.save([train_dataset_splited, test_dataset_o], file_path)

# %% 参数定义
alpha = 0.5

num_all_client = 6
num_all_server = 2
all_client = number_list(num_all_client)
all_server = number_list(num_all_server)
num_server_commu = 2
num_client_commu = 2
num_client_train = 2
num_public_train = 2

server_client_1 = [0, 1, 2]
server_client_2 = [3, 4, 5]
server_client = [server_client_1, server_client_2]
neighbor_server = [[1], [0]]

# %% 训练集分配
idx_client_target = train_data_split(train_dataset_splited, all_client)
all_target = train_dataset_splited.keys()

public_idx, test_idx = split_idx_proportion(
    [i for i in range(len(test_dataset_o))], [0.2, 0.8])
public_dataset = [test_dataset_o[idx] for idx in public_idx]
test_dataset = [test_dataset_o[idx] for idx in test_idx]

train_dataset_client = list_same_term(num_all_client)
for client in all_client:
    for target in all_target:
        train_dataset_client_new = [train_dataset_splited[target][idx]
                                    for idx in idx_client_target[client][target]]
        train_dataset_client[client].extend(train_dataset_client_new)

# %% 模型初始化
initial_model = LeNet5(28, 28, 1, 10)

client_model = list_same_term(num_all_client, initial_model)
server_model = list_same_term(num_all_server)
server_accuracy = list_same_term(num_all_server)
client_accuracy = list_same_term(num_all_client)
server_client_model = list_same_term(num_all_server)
server_model_distillation_accuracy = list_same_term(num_all_server)

# %% 模型训练
# 对每个服务器通讯幕进行循环
for epoch_server_commu in tqdm(range(num_server_commu), desc='epoch_server_commu', position=0, leave=False):
    # 所有边缘服务器分别协调其客户端进行多轮联邦学习
    for epoch_client_commu in tqdm(range(num_client_commu), desc='epoch_client_commu', position=1, leave=False):
        # 所有边缘服务器分别协调其客户端进行联邦学习
        for server in tqdm(all_server, desc='server', position=2, leave=False):
            # 每个服务器下单客户端分别训练
            for client in tqdm(server_client[server], desc='client', position=3, leave=False):
                # 单个服务器下的客户端在私有数据集上进行num_client_train轮训练
                client_model[client], loss_sum = train_model(
                    model=client_model[client],
                    dataset=train_dataset_client[client],
                    device=device,
                    epochs=num_client_train)
                # 单个服务器下的客户端在公开数据集上进行num_public_train轮训练
                if epoch_server_commu != 0:
                    neighbor_server_model = [
                        server_model_distillation[server] for server in neighbor_server[server]]
                    weight = torch.tensor(
                        [1/len(neighbor_server_model) for _ in neighbor_server_model]).to(device)
                    client_model[client], loss_sum = train_model_disti(
                        model=client_model[client],
                        neighbor_server_model=neighbor_server_model,
                        weight=weight,
                        dataset=public_dataset,
                        device=device,
                        epochs=num_public_train,
                        num_target=len(all_target),
                        alpha=alpha)
                # 在训练后评估该服务器下的客户端
                client_accuracy[client].append(eval_model(
                    client_model[client], test_dataset, device))
            # 在单个服务器下客户端训练完成后更新该服务器下客户端的模型
            server_client_model[server] = [client_model[client]
                                           for client in server_client[server]]
            # 聚合获得单个服务器模型
            server_model[server] = EdgeServer(
                server_client_model[server]).average()
            # 评估单个服务器模型
            server_accuracy[server].append(eval_model(server_model[server], test_dataset, device).cpu())
    # 服务器在多轮更新联邦学习后固定用于蒸馏的模型
    server_model_distillation = deepcopy(server_model)
    # 评估该蒸馏模型
    for server in all_server:
        server_model_distillation_accuracy[server].append(eval_model(
            server_model_distillation[server], test_dataset, device).cpu())

# %% 作图
x = [i for i in range(num_server_commu * num_client_commu)]
line_list = []
for server in all_server:
    line_list.append(plt.plot(x, server_accuracy[server]))

plt.show()
