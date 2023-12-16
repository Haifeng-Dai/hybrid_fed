def idx_to_dataset(dataset, idxs):
    '''
    返回对应指标集的数据子集
    '''
    dataset_idex = [dataset[idx] for idx in idxs]
    return dataset_idex

a = [(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)]
idx = [1, 3, 0]
b = idx_to_dataset(a, idx)

print(b)