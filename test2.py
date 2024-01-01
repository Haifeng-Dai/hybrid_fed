import torch
import sys

from mpi4py import MPI
from copy import deepcopy

from utils.model_util import *
from utils.data_util import *
from utils.lib_util import *
from utils.train_util import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
total_rank = comm.Get_size()

file_path = './test.pt'
loaded_data = torch.load(file_path)
args = loaded_data['args']
num_target = loaded_data['num_target']
num_server_client = loaded_data['num_server_client']
train_dataloader = loaded_data['train_dataloader']
test_dataloader = loaded_data['test_dataloader']
train_test_dataloader = loaded_data['train_test_dataloader']
public_dataloader = loaded_data['public_dataloader']
server_client = loaded_data['server_client']
neighbor_server = loaded_data['neighbor_server']
all_server = loaded_data['all_server']
all_client = loaded_data['all_client']
server_model = loaded_data['server_model']
client_model = loaded_data['client_model']
client_accuracy = loaded_data['client_accuracy']
server_accuracy = loaded_data['server_accuracy']
client_loss = loaded_data['client_loss']
log_path = loaded_data['log_path']

log = get_logger(log_path, mode='a')
server_model_ = deepcopy(server_model)

model = client_model[rank]
__acc = eval_model(model=model,
                   dataloader=test_dataloader,
                   device=args.device)
# if rank == 0:
#     for client in client_model:
#         __acc = eval_model(model=client,
#                         dataloader=test_dataloader,
#                         device=args.device)
#         print(__acc)

comm.barrier()
for epoch_server_commu in range(args.num_server_commu):
    if rank == 0:
        log.info('-'*50)
        log.info('|epoch_server_commu: {}/{}'. format(epoch_server_commu +
                 1, args.num_server_commu))
    # 所有边缘服务器分别协调其客户端进行多轮联邦学习
    for epoch_client_commu in range(args.num_client_commu):
        if rank == 0:
            message = ' |epoch_client_commu: {}/{}'.format(
                epoch_client_commu + 1, args.num_client_commu)
            log.info(message)
        # 客户端训练
        loss_ = []
        acc_ = []
        comm.barrier()
        if rank == 0:
            for client in all_client:
                __acc = eval_model(model=client_model[client],
                                dataloader=test_dataloader,
                                device=args.device)
                print('before', __acc)
        comm.barrier()
        for epoch_client_train in range(args.num_client_train):
            model, loss__ = train_model(model=model,
                                        dataloader=train_dataloader[rank],
                                        device=args.device)
            loss_.extend(loss__)
            acc__ = eval_model(model=model,
                               dataloader=test_dataloader,
                               device=args.device)
            acc_.append(acc__)
            message = '  |epoch_client: {}/{}, client: {}/{}, loss: {:.4f}, acc: {:.4f}'.format(
                epoch_client_train + 1, args.num_client_train, rank, total_rank, loss__[-1], acc__)
            log.info(message)
        comm.barrier()
        if rank == 0:
            __acc = eval_model(model=model,
                            dataloader=test_dataloader,
                            device=args.device)
            print('center1', __acc)
        comm.barrier()
        # for server, servers in enumerate(neighbor_server):
        #     if rank in server_client[server]:
        #         neighbors = deepcopy(servers)
        #         break
        # if epoch_server_commu != 0:
        #     for server in neighbors:
        #         for epoch_public_train in range(args.num_public_train):
        #             model, loss__ = train_model_disti_single(
        #                 model=model,
        #                 teacher_model=server_model[server],
        #                 dataloader=public_dataloader,
        #                 alpha=args.alpha,
        #                 T=args.T,
        #                 device=args.device)
        #             loss_.extend(loss__)
        #             acc__ = eval_model(model=model,
        #                                dataloader=test_dataloader,
        #                                device=args.device)
        #             acc_.append(acc__)
        #             message = '  |epoch_distill: {}/{}, client: {}/{}, server: {}, loss: {:.4f}, acc: {:.4f}'.format(
        #                 epoch_client_train + 1, args.num_client_train, rank, total_rank, server, loss__[-1], acc__)
        #             log.info(message)
        # comm.barrier()
        # if rank == 0:
        #     __acc = eval_model(model=model,
        #                     dataloader=test_dataloader,
        #                     device=args.device)
        #     print('center2', __acc)
        # comm.barrier()
        # gather model
        client_model_ = comm.gather(model, root=0)
        if rank == 0:
            for client in client_model_:
                __acc = eval_model(model=client,
                                dataloader=test_dataloader,
                                device=args.device)
                print('after', __acc)
        comm.barrier()
        # # gather loss and accuracy
        # loss = comm.gather(loss_, root=0)
        # acc = comm.gather(acc_, root=0)
        # # 聚合
        # comm.barrier
        # if rank == 0:
        #     # aggregate loss and accuracy
        #     for client in all_client:
        #         client_loss[client].extend(loss[client])
        #         client_accuracy[client].extend(acc[client])
        #     # server weighted aggregate
        #     for server, clients in enumerate(server_client):
        #         # print(clients)
        #         clients_model = [deepcopy(client_model_[i]) for i in clients]
        #         server_model_[server] = aggregate(
        #             clients_model, [0.2, 0.3, 0.5])
        #         server_acc = eval_model(model=server_model_[server],
        #                                 dataloader=test_dataloader,
        #                                 device=args.device)
        #         server_accuracy[server].append(server_acc)
        #         for client in clients:
        #             # print(client)
        #             client_model[client] = deepcopy(server_model_[server])
        #         message = ' |epoch_client_commu: {}/{}, server aggregated, server {} acc: {}'.format(
        #             epoch_client_commu + 1, args.num_client_commu, server, server_acc)
        #         # log.info(message)
        #     server_model = deepcopy(server_model_)
        #     for client in client_model:
        #         __acc = eval_model(model=client,
        #                            dataloader=test_dataloader,
        #                            device=args.device)
        #         print('agg', __acc)
        # comm.barrier()
        # client_model = comm.bcast(client_model, root=0)
        # comm.barrier()
    # server communication
#     comm.barrier()
#     if args.algorithm == 2 or args.algorithm == 3:
#         server_model = comm.bcast(server_model, root=0)
#         message = 'server communication {}/{}'.format(
#             epoch_server_commu, args.num_server_commu)
#     elif args.algorithm == 4:
#         if rank == 0:
#             weight_list = list_same_term(3, [1/3, 1/3, 1/3])
#             server_model = server_communicate(server_model, weight_list)
#             for server in all_server:
#                 for client in server_client[server]:
#                     client_model[client] = deepcopy(server_model[server])
#             client_model = comm.bcast(client_model, root=0)
#             message = 'server communication {}/{}'.format(
#                 epoch_server_commu, args.num_server_commu)
#     else:
#         message = 'algorithm {}, no server communication.'.format(
#             args.algorithm)
#         log.info(message)
#     comm.barrier()

# # %% 保存
# save_data = {'args': args,
#              #  'server_model': server_model_save,
#              'server_acc': server_accuracy,
#              #  'client_model': client_model_save,
#              'client_acc': client_accuracy,
#              #  'train_acc': train_accuracy,
#              'client_loss': client_loss}
# save_file(args, save_data, log, rank)
