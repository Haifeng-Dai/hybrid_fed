import torchvision
import torch

from torch.utils.data import DataLoader, Dataset
from utils.model_util import LeNet5, CNN, MLP
from utils.train_util import *

torch.set_printoptions(precision=2,
                       threshold=1000,
                       edgeitems=5,
                       linewidth=1000,
                       sci_mode=False)

dataset = torchvision.datasets.MNIST(
    root='./data/raw-data',
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False)

model = CNN(28, 28, 1, 10)
trained_model = train_model(
    model=model,
    device='cuda',
    dataset=dataset,
    criterion=torch.nn.CrossEntropyLoss())

