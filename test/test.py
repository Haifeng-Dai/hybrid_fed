import torchvision
import torch

from torch.utils.data import DataLoader, Dataset
from utils.model_util import LeNet5, CNN, MLP
from utils.train_util import *

model = LeNet5(28, 28, 1, 10).cuda()
train_dataset_loader = DataLoader(
    dataset=torchvision.datasets.MNIST(
        root='../data/raw-data/',
        train=True,
        transform=torchvision.transforms.ToTensor()),
    batch_size=250,
    shuffle=True)
test_dataset_loader = DataLoader(
    dataset=torchvision.datasets.mnist.MNIST(
        root='../data/raw-data/',
        transform=torchvision.transforms.ToTensor()),
    batch_size=32,
    shuffle=False)

opti = torch.optim.Adam(model.parameters())
criterion = torch.nn.CrossEntropyLoss()
for data, target in train_dataset_loader:
    opti.zero_grad()
    output = model(data.cuda())
    loss = criterion(output, target.cuda())
    loss.backward()
    opti.step()
    break

print(type(loss), loss)

model_copy = deepcopy(model).eval()
model_copy.eval()
model_copy.cuda()
correct = 0
total = 0
data_loader = test_dataset_loader
for images, targets in data_loader:
    outputs = model_copy(images.cuda())
    _, predicted = torch.max(outputs.data, 1)
    total += targets.size(0)
    correct += (predicted == targets.cuda()).sum().item()
    break
print('Test Accuracy: {:.2f}%'.format(100 * correct / total))