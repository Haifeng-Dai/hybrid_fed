import sys
import os
import logging
import torch
import argparse

import matplotlib.pyplot as plt

from copy import deepcopy


def list_same_term(len_list, term=[]):
    # 返回一个全是空列表的列表
    list_return = []
    for _ in range(len_list):
        list_return.append(deepcopy(term))
    return list_return


# def number_list(len_list):
#     '''
#     返回一个数字列表
#     '''
#     return [i for i in range(len_list)]


# def split_idx_evenly(idxs, num_set):
#     '''
#     把给定的指标集随机分割成几个集合
#     input: index list
#     out put: a list with indexes of each set
#     '''
#     idxs_copy = deepcopy(idxs)
#     random.shuffle(idxs_copy)
#     idx_cut = idxs_copy[ : len(idxs_copy) // num_set * num_set]
#     idx_numpy = numpy.array(idx_cut)
#     idx_set_matrix = idx_numpy.reshape(num_set, -1)
#     idx_set = idx_set_matrix.tolist()
#     return idx_set


# def split_idx_proportion(idx, proportion):
#     '''
#     把给定指标集按给定比例分割
#     input: two lists
#     output: a list
#     '''
#     num_set = len(proportion)
#     num_idx_set = list()
#     for set in range(num_set):
#         num_idx_set.append(round(len(idx) * proportion[set]))
#     idx_copy = deepcopy(idx)
#     random.shuffle(idx_copy)
#     idx_sets = []
#     idx_left = idx_copy
#     for set in range(num_set):
#         idx_set = idx_left[: num_idx_set[set]]
#         idx_left = idx_left[num_idx_set[set]:]
#         idx_sets.append(idx_set)
#     return idx_sets


def get_logger(filename, mode='w'):
    # 日志设置
    log_formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S')
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    # File logger
    file_handler = logging.FileHandler(filename, mode=mode)
    file_handler.setFormatter(log_formatter)
    file_handler.setLevel(logging.DEBUG)
    logger.addHandler(file_handler)
    # Stdout logger
    std_handler = logging.StreamHandler(sys.stdout)
    std_handler.setFormatter(log_formatter)
    std_handler.setLevel(logging.DEBUG)
    logger.addHandler(std_handler)
    return logger


def save_file(args, save_data, log):
    # 保存数据
    save_path = f'./res/{args.dataset}_model_{args.model_select}_algo_{args.algorithm}/alpha_{args.alpha}_T_{args.T}/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    file_path = save_path+(
        f'server_commu_{args.num_server_commu}'
        f'_client_commu_{args.num_client_commu}'
        f'_client_train_{args.num_client_train}'
        f'_batch_size_{args.batch_size}'
        f'_num_all_client_{args.num_all_client}'
        f'_num_all_server_{args.num_all_server}'
        f'_num_client_data_{args.num_client_data}'
        f'_num_public_data_{args.num_public_data}'
        f'_proportion_{args.proportion}.pt')
    log.info(file_path)
    torch.save(save_data, file_path)


def get_args():
    # 获取输入参数
    parser = argparse.ArgumentParser(description='save results.')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='the used dataset.')
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='trade-off parameters of distillation.')
    parser.add_argument('--T', type=int, default=2.0,
                        help='temperature of distillation.')
    parser.add_argument('--num_all_client', type=int,
                        default=9, help='num of all client.')
    parser.add_argument('--num_all_server', type=int,
                        default=3, help='num of all server.')
    parser.add_argument('--batch_size', type=int, default=200,
                        help='batch size of trainning.')
    parser.add_argument('--num_client_data', type=int,
                        default=1000, help='number of client datas.')
    parser.add_argument('--num_server_commu', type=int, default=1,
                        help='number of server communications.')
    parser.add_argument('--num_client_commu', type=int, default=1,
                        help='number of clients communicate with servers.')
    parser.add_argument('--num_client_train', type=int, default=1,
                        help='number of client train in local data.')
    parser.add_argument('--num_public_train', type=int, default=1,
                        help='number of client distillation in public data.')
    parser.add_argument('--model_select', type=int, default=1,
                        help='select the model group.')
    parser.add_argument('--algorithm', type=int, default=1,
                        help='select the algorithm.')
    # parser.add_argument('--graph', type=int, default=1,
    #                     help='select the graph.')
    parser.add_argument('--num_public_data', type=int, default=50,
                        help='number of public data.')
    parser.add_argument('--proportion', type=float, default=0.8,
                        help='proportion of main target data.')
    return parser.parse_args()


# if __name__ == '__main__':
#     import time
#     t = time.localtime()
#     a = 1
#     b = 2
#     c = ' '*20
#     massage = '{}:{:^5}\n{}{}:{:^5}'.format('a', a, ' '*29, 'b', b)
#     file_name = f'./log/{t.tm_year}-{t.tm_mon}-{t.tm_mday}-{t.tm_hour}-{t.tm_min} test.txt'

#     logger = get_logger(file_name)
#     logger.info(massage)

    # # deepcopy is importent
    # a = list_same_term(2)
    # print('a', a)
    # for i, _ in enumerate(a):
    #     a[i].append(i)
    # print('a', a)
    # b = list_same_term(3)
    # print('b', b)

    # bb = list_same_term(3, b)
    # print(bb)
    # bb[0][0] = 1
    # print(bb)

    # c = number_list(3)
    # print(c)
    # d = number_list(3)
    # d[0] = 1
    # print(d)
