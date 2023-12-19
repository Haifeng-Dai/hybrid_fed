from copy import deepcopy

def get_list(len_list, term=[]):
    '''
    返回一个全是空列表的列表
    '''
    list_return = []
    # if not term:
    #     term = []
    for _ in range(len_list):
        list_return.append(deepcopy(term))
    return list_return


if __name__ == '__main__' :
    # deepcopy is importent
    a = get_list(2)
    print(a)
    for i, _ in enumerate(a):
        a[i].append(i)
    print(a)
    b = get_list(3)
    print(b)