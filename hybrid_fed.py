import torch
import os

from torch.utils.data import DataLoader

from utils.model_util import LeNet5, CNN
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

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)

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

alpha = 0.8
beta = 0.2

num_all_client = 6
num_all_server = 2
all_client = [i for i in range(num_all_client)]
all_server = [i for i in range(num_all_server)]
num_server_commu = 2
num_client_commu = 2
num_client_train = 2
num_public_train = 2

server_client_1 = [0, 1, 2]
server_client_2 = [3, 4, 5]
server_client = [server_client_1, server_client_2]
neighbor_server = [1, 0]

idx_client_target = train_data_split(train_dataset_splited, all_client)
all_target = train_dataset_splited.keys()

public_idx, test_idx = split_idx_proportion([i for i in range(len(test_dataset_o))], [0.2, 0.8])
public_dataset = [test_dataset_o[idx] for idx in public_idx]
test_dataset = [test_dataset_o[idx] for idx in test_idx]

train_dataset_client = get_list(num_all_client)
for client in all_client:
    for target in all_target:
        train_dataset_client_new = [train_dataset_splited[target][idx] for idx in idx_client_target[client][target]]
        train_dataset_client[client].extend(train_dataset_client_new)

initial_model = LeNet5(28, 28, 1, 10)

client_model = get_list(num_all_client, initial_model)
model_server = get_list(num_all_server)

for epoch_server_commu in range(num_server_commu):
    for epoch_client_commu in range(num_client_commu):
        # 边缘服务器协调其客户端进行联邦学习
        for server in all_server:
            for client in server_client[server]:
                # 在私有数据集上进行训练
                client_model[client] = train_model(
                    model=client_model[client],
                    dataset=train_dataset_client[client],
                    device=device,
                    epochs=num_client_train)
                if epoch_server_commu != 0:
                    # 在public数据集上进行训练
                    neighbor_server_model = [client_model[client] for client in server_client[neighbor_server[server]]]
                    weight = torch.tensor([1/len(neighbor_server_model) for _ in neighbor_server_model]).to(device)
                    client_model[client] = train_model_disti(
                        model=client_model[client],
                        neighbor_server_model = neighbor_server_model,
                        weight= weight,
                        dataset=public_dataset,
                        device=device,
                        epochs=num_public_train,
                        num_target=len(all_target),
                        alpha=alpha,
                        beta=beta)
            for server in all_server:
                server_client_model = [client_model[client]
                                    for client in server_client[server]]
                model_server[server] = EdgeServer(server_client_model).average()

the_model = deepcopy(model_server[server]).eval().to(device)
# eval_model(the_model, test_dataset, device)
test_dataloader = DataLoader(test_dataset, 1)
for data, _ in test_dataloader:
    output = the_model(data.to(device))
    print(output.size())
    print(output[0].tolist())
    break