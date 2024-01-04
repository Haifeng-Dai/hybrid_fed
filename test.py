import torch

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
device = 'cuda'

train_dataset_o, test_dataset_o, c, h, w = get_dataset('cifar10')
len_train_dataset = int(len(train_dataset_o))
print(len_train_dataset, int(len(test_dataset_o)), c, h, w)
# [public_dataset, test_dataset] = split_parts_random(
#     train_dataset_o, [1000, len_train_dataset - 1000])
# train_dataloader = DataLoader(dataset=public_dataset,
#                               batch_size=32,
#                               shuffle=True)
test_dataloader = DataLoader(dataset=test_dataset_o,
                             batch_size=1000)

TrainDatasetSplited = SplitData(train_dataset_o)
all_target = TrainDatasetSplited.targets
num_target = TrainDatasetSplited.num_target

client_main_target = numpy.random.choice(
    all_target, 2, replace=False).tolist()
train_dataset_client = TrainDatasetSplited.server_non_iid(
    num_server=2,
    num_server_client=2,
    num_client_data=1200,
    client_main_target=client_main_target,
    proportion=0.8)
[dataset_train, dataset_test] = split_parts_random(
    train_dataset_client[0], [1000, 200])
train_test_dataloader = DataLoader(dataset=dataset_test,
                                   batch_size=32,
                                   shuffle=True)
train_dataloader = DataLoader(dataset=dataset_train,
                              batch_size=32,
                              shuffle=True)

model = CNN(h, w, c, 100).to(device)
for epoch in range(100):
    model, loss = train_model(model=model,
                              dataloader=train_dataloader,
                              device=device,
                              LR=1e-3)

    acc1 = eval_model(model=model,
                      dataloader=train_test_dataloader,
                      device=device)

    acc2 = eval_model(model=model,
                      dataloader=test_dataloader,
                      device=device)

    print(acc1, acc2)
