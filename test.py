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

model = CNN(h, w, c, 10).to(device)

train_dataloader = DataLoader(dataset=train_dataset_o,
                              batch_size=160,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_dataset_o,
                             batch_size=1000)

optimizer = torch.optim.Adam(params=model.parameters())
loss_ = []
for epoch in range(10):
    for data, target in train_dataloader:
        optimizer.zero_grad()
        output = model(data.to(device))
        loss = torch.nn.functional.cross_entropy(output, target.to(device))
        loss.backward()
        optimizer.step()
        loss_.append(loss.item())

    acc = eval_model(model=model,
                     dataloader=test_dataloader,
                     device=device)

    print(acc)
