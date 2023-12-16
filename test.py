import random
import copy

a = [x for x in range(10)]
b = copy.deepcopy(a)
random.shuffle(b)
print(a)
print(b)