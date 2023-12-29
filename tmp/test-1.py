from utils.data_util import *

train_dataset, test_dataset, c, h, w = get_dataset()


def data_loader(data_set, batch_size, dataset='mnist', shuffle=False, device='cpu'):
    num_data = len(data_set)
    if dataset == 'mnist':
        images = data_set.data.unsqueeze(1) / 255
        target = data_set.targets.int()
        # images = [torch.from_numpy(numpy.array(data[0])).float()[None, None, :, :] for data in data_set]
        # target = data_set.targets.int()
    elif dataset == 'cifar10':
        images = data_set.data.transpose([0, 3, 1, 2])
        cifar10_train_mean = torch.tensor((0.4914, 0.4822, 0.4465))[
            None, :, None, None]
        cifar10_train_std = torch.tensor((0.2470, 0.2435, 0.2616))[
            None, :, None, None]
        images = torch.tensor(images, dtype=torch.int32) - cifar10_train_mean
        images /= cifar10_train_std
        target = torch.tensor(data_set.targets, dtype=torch.int32)
    idxs = [i for i in range(num_data)]
    if shuffle:
        random.shuffle(idxs)
    num_dataloader = num_data // batch_size
    if num_data % batch_size:
        num_dataloader += 1
    for i in range(num_dataloader):
        idx = i * batch_size
        data_return = images[idx: idx+batch_size]
        target_return = target[idx: idx+batch_size]
        yield (data_return.to(device), target_return.to(device))


# %% intial
import torch
import numpy
import time
import sys

from torch.utils.data import DataLoader

from utils.model_util import *
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
elif torch.backends.mps.is_available():
    device = 'mps'
else:
    device = 'cpu'

dataset = 'mnist'
# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()
# alpha = 0.5
# T = 2
# num_server_commu = 15
# num_client_commu = 10
# num_client_train = 10
# num_public_train = 10
# batch_size = 200

num_all_client = 9
num_all_server = 3
num_client_data = 1000
# num_public_data = 50
proportion = 0.8
# server_client = [[0, 1, 2], [3, 4, 5], [6, 7, 8]]
# neighbor_server = [[1], [2], [0]]
num_server_client = num_all_client // num_all_server

train_dataset_o, test_dataset_o, c, h, w = get_dataset(dataset)
TrainDatasetSplited = SplitData(train_dataset_o)
all_target = TrainDatasetSplited.targets
num_target = TrainDatasetSplited.num_target

# print(list(all_target))
client_main_target = numpy.random.choice(
    all_target, num_all_client, replace=False).tolist()
train_dataset_client = TrainDatasetSplited.server_non_iid(
    num_server=num_all_server,
    num_server_client=num_server_client,
    num_client_data=num_client_data,
    client_main_target=client_main_target,
    proportion=proportion)