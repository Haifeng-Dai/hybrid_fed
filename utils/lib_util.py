import random
import numpy

from copy import deepcopy


def empty_list(len_list, term=[]):
    '''
    返回一个全是空列表的列表
    '''
    list_return = []
    for _ in range(len_list):
        list_return.append(deepcopy(term))
    return list_return


def number_list(len_list):
    '''
    返回一个数字列表
    '''
    return [i for i in range(len_list)]


def split_idx_evenly(idxs, num_set):
    '''
    把给定的指标集随机分割成几个集合
    input: index list
    out put: a list with indexes of each set
    '''
    idxs_copy = deepcopy(idxs)
    random.shuffle(idxs_copy)
    idx_cut = idxs_copy[ : len(idxs_copy) // num_set * num_set]
    idx_numpy = numpy.array(idx_cut)
    idx_set_matrix = idx_numpy.reshape(num_set, -1)
    idx_set = idx_set_matrix.tolist()
    return idx_set


def split_idx_proportion(idx, proportion):
    '''
    把给定指标集按给定比例分割
    input: two lists
    output: a list
    '''
    num_set = len(proportion)
    num_idx_set = list()
    for set in range(num_set):
        num_idx_set.append(round(len(idx) * proportion[set]))
    idx_copy = deepcopy(idx)
    random.shuffle(idx_copy)
    idx_sets = []
    idx_left = idx_copy
    for set in range(num_set):
        idx_set = idx_left[ : num_idx_set[set]]
        idx_left = idx_left[num_idx_set[set] : ]
        idx_sets.append(idx_set)
    return idx_sets


if __name__ == '__main__':
    # deepcopy is importent
    a = empty_list(2)
    print(a)
    for i, _ in enumerate(a):
        a[i].append(i)
    print(a)
    b = empty_list(3)
    print(b)

    c = number_list(3)
    print(c)
    d = number_list(3)
    d[0] = 1
    print(d)
