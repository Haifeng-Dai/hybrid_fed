import torch
import torchvision
import numpy
import copy
import tqdm

from utils.model import LeNet5
from utils.lib import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    device = 'cuda'
else:
    device = 'cpu'
print(device)

train_dataset, test_dataset = get_dataset(dataset='mnist')

num_all_client = 5
num_data = 12000
communication_round = 1
epochs = 1
num_client = 5

model = LeNet5(28, 28, 1, 10)
idx_splited = idx_split(
    dataset=train_dataset,
    n_dataset=num_all_client,
    n_data_each_set=num_data
)
dataset_client = dict()
for i in range(num_all_client):
    dataset_client[i] = DealDataset(train_dataset, idx_splited[i])

server_model = copy.deepcopy(model)
tqdm_position = 0
for i in range(communication_round):
    client = dict()
    client_param = dict()
    choicen_client = numpy.random.choice(
        range(num_all_client), num_client, replace=False)
    for j, k in enumerate(choicen_client):
        client[j] = train_model(
            model=server_model,
            dataset=dataset_client[k],
            device=device,
            epochs=epochs,
            tqdm_position=tqdm_position+1
        )
        client_param[j] = client[j].state_dict()
    server_model1 = Server(model, client_param).average()
    server_model2 = Server(model, client_param).weighted_average(
        weight=[0.1, 0.15, 0.2, 0.25, 0.3])

for i in range(num_all_client):
    eval_model(client[i], test_dataset, device)
eval_model(server_model1, test_dataset, device)
eval_model(server_model2, test_dataset, device)
eval_model(server_model, test_dataset, device)
eval_model(model, test_dataset, device)
