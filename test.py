import torch
import torchvision

net1 = torchvision.models.resnet18(weights=None, num_classes=10)
net2 = torchvision.models.resnet34(weights=None, num_classes=10)
net3 = torchvision.models.resnet50(weights=None, num_classes=10)
net4 = torchvision.models.resnet101(weights=None, num_classes=10)
net5 = torchvision.models.resnet152(weights=None, num_classes=10)
net6 = torchvision.models.vgg11_bn(weights=None, num_classes=10)
print(len(net1.state_dict().keys()))
print(len(net2.state_dict().keys()))
print(len(net3.state_dict().keys()))
print(len(net4.state_dict().keys()))
print(len(net5.state_dict().keys()))
print(len(net6.state_dict().keys()))
