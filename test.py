import numpy
import torchvision

a = numpy.random.random([32, 34])

b = torchvision.transforms.ToTensor()(a)

print(a, a.shape)
print(b, b.shape)