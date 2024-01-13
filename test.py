import torch
import torchvision

from utils.data_util import *
from utils.train_util import *
from torch.utils.data import DataLoader

net1 = torchvision.models.resnet18(weights=None, num_classes=10)
gpus = [0, 1, 2]
model = torch.nn.DataParallel(net1.cuda(), device_ids=gpus, output_device=gpus[0])

train_dataset, test_dataset, c, h, w = get_dataset('cifar10')
trainloader = DataLoader(
    dataset=train_dataset,
    batch_size=320,
    shuffle=True,
    pin_memory=True,
    num_workers=8)
testloader = DataLoader(
    dataset=test_dataset,
    batch_size=160,
    shuffle=True,
    pin_memory=True,
    num_workers=8)


for i in range(10):
    trained_model = deepcopy(model)
    trained_model.train()
    optimizer = torch.optim.Adam(trained_model.parameters())
    loss_ = []
    for data, target in trainloader:
        optimizer.zero_grad()
        output = trained_model(data)
        loss = torch.nn.CrossEntropyLoss()(output, target.cuda())
        loss.backward()
        optimizer.step()
        loss_.append(loss.item())
    print(i)
