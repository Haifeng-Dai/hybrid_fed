def get_list(len_list, term=[]):
    '''
    返回一个全是空列表的列表
    '''
    list_return = []
    for _ in range(len_list):
        list_return.append(term)
    return list_return