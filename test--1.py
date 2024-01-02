import torch

from copy import deepcopy
from utils.model_util import *

model1 = CNN(28, 28, 1, 10)
model2 = CNN(28, 28, 1, 10)

layer_name = model1.state_dict().keys()
print(len(layer_name), list(layer_name)[0], layer_name)

params_model1 = model1.state_dict()
params_model2 = model2.state_dict()

model3 = deepcopy(model1)
model4 = deepcopy(model2)

for key in layer_name:
    layer_params = model1
