import torch

from torch.utils.data import DataLoader, Dataset
from utils.model_util import LeNet5
from utils.data_util import *
from utils.train_util import eval_model


class Loss1(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        return torch.nn.functional.cross_entropy(input, target)


def loss2(x, y):
    return torch.mean(torch.pow((x - y), 2))


device = 'cuda'
model = LeNet5(28, 28, 1, 10)
train_dataset, test_dataset = get_dataset()
train_dataloader = DataLoader(train_dataset, 32, True)

trained_model = copy.deepcopy(model).to(device)
trained_model.train()
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
# ce_loss = torch.nn.CrossEntropyLoss()
# kl_loss = torch.nn.KLDivLoss(reduction='batchmean')
my_loss = Loss1()
optimizer = torch.optim.Adam(trained_model.parameters())
for i, (data, label) in enumerate(train_dataloader):
    optimizer.zero_grad()
    output = trained_model(data.to(device))
    loss = my_loss(output, label.to(device))
    loss.backward()
    optimizer.step()

eval_model(trained_model, test_dataset, device)
