import torchvision
import torch

from torch.utils.data import DataLoader, Dataset

torch.set_printoptions(precision=2,
                       threshold=1000,
                       edgeitems=5,
                       linewidth=1000,
                       sci_mode=False)

a = torch.randn([3, 4])
print(a)

b = torch.flatten(a)
print(b)
print(b.shape)
print(b.shape[0])