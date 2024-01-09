import sys
import os
import logging
import torch
import argparse

import matplotlib.pyplot as plt

from copy import deepcopy
from utils.train_util import *


def list_same_term(len_list, term=[]):
    # 返回一个全是空列表的列表
    list_return = []
    for _ in range(len_list):
        list_return.append(deepcopy(term))
    return list_return


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


def aggregate(model_list, weight):
    aggregated_model = deepcopy(model_list[0])
    parameters = deepcopy(model_list[0].state_dict())
    for i, key in enumerate(parameters):
        if parameters[key].shape == torch.Size([]):
            continue
        parameters[key] *= weight[0]
    for i, model in enumerate(model_list[1:]):
        for key in parameters:
            if parameters[key].shape == torch.Size([]):
                continue
            parameters[key] += model.state_dict()[key] * weight[i+1]
    aggregated_model.load_state_dict(parameters)
    return aggregated_model


def cal_gp(D, real_imgs, fake_imgs, device):  # 定义函数，计算梯度惩罚项gp
    # 真假样本的采样比例r，batch size个随机数，服从区间[0,1)的均匀分布
    r = torch.rand(size=(real_imgs.shape[0], 1, 1, 1)).to(device)
    # 输入样本x，由真假样本按照比例产生，需要计算梯度
    x = (r * real_imgs + (1 - r) * fake_imgs).requires_grad_(True)
    d = D(x)  # 判别网络D对输入样本x的判别结果D(x)
    fake = torch.ones_like(d).to(device)  # 定义与d形状相同的张量，代表梯度计算时每一个元素的权重
    g = torch.autograd.grad(  # 进行梯度计算
        outputs=d,  # 计算梯度的函数d，即D(x)
        inputs=x,  # 计算梯度的变量x
        grad_outputs=fake,  # 梯度计算权重
        create_graph=True,  # 创建计算图
        retain_graph=True  # 保留计算图
    )[0]  # 返回元组的第一个元素为梯度计算结果
    gp = ((g.norm(2, dim=1) - 1) ** 2).mean()  # (||grad(D(x))||2-1)^2 的均值
    return gp  # 返回梯度惩罚项gp


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)
