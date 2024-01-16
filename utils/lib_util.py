import sys
import os
import logging
import torch
import argparse
import torchvision

from copy import deepcopy
from utils.train_util import *
from utils.model_util import *


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
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num_workers.')
    parser.add_argument('--num_public_data', type=int, default=50,
                        help='number of public data.')
    parser.add_argument('--proportion', type=float, default=0.8,
                        help='proportion of main target data.')
    return parser.parse_args()


def get_device(log):
    if torch.cuda.is_available():
        device = 'cuda'
        log.info(f'device {device} is used.')
        # num_device = torch.cuda.device_count()
        # if num_device > 1:
        #     device = [i for i in range(num_device)]
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
            log.info('cudnn is actived.')
    elif torch.backends.mps.is_available():
        device = 'mps'
        log.info(f'device {device} is used.')
    else:
        device = 'cpu'
        log.info(f'device {device} is used.')
    return device


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
    os.makedirs(save_path, exist_ok=True)
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


def intial_model(args, num_target, num_server_client, c, h, w):
    if args.model_select == 1:
        model = CNN(h, w, c, num_target)
        client_model = list_same_term(args.num_all_client, model)
        server_model = list_same_term(args.num_all_server, model)
    elif args.model_select == 2:
        model = LeNet5(h, w, c, num_target)
        client_model = list_same_term(args.num_all_client, model)
        server_model = list_same_term(args.num_all_server, model)
    elif args.model_select == 3:
        model = torchvision.models.resnet18(
            weights=None, num_classes=num_target)
        client_model = list_same_term(args.num_all_client, model)
        server_model = list_same_term(args.num_all_server, model)
    elif args.model_select == 4:
        model1 = CNN(h, w, c, num_target)
        model2 = LeNet5(h, w, c, num_target)
        model3 = torchvision.models.resnet18(weights=None, num_classes=10)
        server_model = [model1, model2, model3]
        client_model1 = list_same_term(num_server_client, model1)
        client_model2 = list_same_term(num_server_client, model2)
        client_model3 = list_same_term(num_server_client, model3)
        client_model = client_model1 + client_model2 + client_model3
    else:
        raise ValueError('model error.')
    return client_model, server_model


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        torch.nn.init.normal_(m.weight, 1.0, 0.02)
        torch.nn.init.zeros_(m.bias)


def gradient_penality(D, real_samples, fake_samples, device):
    # Calculates the gradient penalty loss for WGAN-GP
    alpha = torch.rand((real_samples.shape[0], 1, 1, 1), device=device)
    interpolates = (
        alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones_like(d_interpolates)
    gradient = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    gradient_ = gradient.view(gradient.shape[0], -1)
    gradient_penalty = ((gradient_.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty


# def gradient_penality(critic, real, fake, device='cpu'):
#     """
#     :param critic: 判别器模型
#     :param real: 真实样本
#     :param fake: 生成的样本
#     :param device: 设备CUP or GPU
#     :return:
#     """
#     BATCH_SIZE, C, H, W = real.shape
#     alpha = torch.randn(size=(BATCH_SIZE, 1, 1, 1)
#                         ).repeat(1, C, H, W).to(device)
#     print(alpha.shape)
#     interpolated_images = real*alpha + fake*(1-alpha)

#     # 计算判别器输出
#     mixed_scores = critic(interpolated_images)
#     # 求导
#     gradient = torch.autograd.grad(
#         inputs=interpolated_images,
#         outputs=mixed_scores,
#         grad_outputs=torch.ones_like(mixed_scores),
#         create_graph=True,
#         retain_graph=True
#     )[0]
#     gradient = gradient.view(gradient.shape[0], -1)
#     gradient_norm = gradient.norm(2, dim=1)
#     gradient_penality = torch.mean((gradient_norm - 1)**2)
#     return gradient_penality
