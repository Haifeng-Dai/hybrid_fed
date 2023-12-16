import numpy
import torchvision

raw_data = './data/raw-data'
train_dataset = torchvision.datasets.MNIST(
    root=raw_data,
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)
test_dataset = torchvision.datasets.MNIST(
    root=raw_data,
    train=False,
    transform=torchvision.transforms.ToTensor()
)

print(train_dataset.train_labels)