import torch
import torchvision
import numpy
import copy
import tqdm

from utils.model_util import LeNet5
from utils.data_util import *
from utils.lib import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)


train_dataset, test_dataset = get_dataset(dataset='mnist')
train_dataset_splited = split_data(train_dataset)

num_all_client = 10
num_all_server = 2
server_commu_round = 3
client_commu_round = 3
client_train_round = 3

server_1_client = [0, 1, 2, 3, 4]
server_2_client = [5, 6, 7, 8, 9]

dataset_client = []
for client in range(num_all_client):
    dataset_client.append([])
for target in train_dataset_splited.keys():
    num_idx_dataset_target = len(train_dataset_splited[target])
    idxs_dataset_target = [i for i in range(num_idx_dataset_target)]
    dataset_target_client = split_idx_evenly(
        idxs_dataset_target, num_all_client)
    for client in range(num_all_client):
        dataset_client[client].extend([train_dataset_splited[client]])

print(type(dataset_client))

# initial_model = LeNet5(28, 28, 1, 10)

# idx_splited = idx_split(
#     dataset=train_dataset,
#     n_dataset=num_all_client,
#     n_data_each_set=num_data
# )
# dataset_client = dict()
# for i in range(num_all_client):
#     dataset_client[i] = DealDataset(train_dataset, idx_splited[i])

# server_model = copy.deepcopy(model)
# tqdm_position = 0
# for i in range(communication_round):
#     client = dict()
#     client_param = dict()
#     choicen_client = numpy.random.choice(
#         range(num_all_client), num_client, replace=False)
#     for j, k in enumerate(choicen_client):
#         client[j] = train_model(
#             model=server_model,
#             dataset=dataset_client[k],
#             device=device,
#             epochs=epochs,
#             tqdm_position=tqdm_position+1
#         )
#         client_param[j] = client[j].state_dict()
#     server_model1 = EdgeServer(model, client_param).average()
#     server_model2 = EdgeServer(model, client_param).weighted_average(
#         weight=[0.1, 0.15, 0.2, 0.25, 0.3])

# for i in range(num_all_client):
#     eval_model(client[i], test_dataset, device)
# eval_model(server_model1, test_dataset, device)
# eval_model(server_model2, test_dataset, device)
# eval_model(server_model, test_dataset, device)
# eval_model(model, test_dataset, device)
