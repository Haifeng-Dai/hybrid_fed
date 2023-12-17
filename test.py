from copy import deepcopy
from utils.model_util import LeNet5

model1 = LeNet5(28, 28, 1, 10)
model2 = LeNet5(28, 28, 1, 10)

model_param1 = model1.state_dict()
model_param2 = model1.state_dict()

print(id(model_param1))
print(id(model_param2))