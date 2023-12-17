import copy
import random

import numpy


def split_idx_evenly(idxs, num_set):
    '''
    把给定的指标集随机分割成几个集合
    input: index list
    out put: a list with indexes of each set
    '''
    idxs_copy = copy.deepcopy(idxs)
    random.shuffle(idxs_copy)
    idx_cut = idxs_copy[:len(idxs_copy)//num_set * num_set]
    idx_numpy = numpy.array(idx_cut)
    idx_set_matrix = idx_numpy.reshape(num_set, -1)
    idx_set = idx_set_matrix.tolist()
    return idx_set

a = [i for i in range(101)]
num_set = 10
c = split_idx_evenly(a, num_set)
print(len(c))
print(a, '\n', c)
a.extend(c[0])
print(a)
