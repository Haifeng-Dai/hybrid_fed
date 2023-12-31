# %% intial
import torch
from mpi4py import MPI

from utils.model_util import *
from utils.data_util import *
from utils.lib_util import *
from utils.train_util import *

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

file_path = './test.pt'
loaded_data = torch.load(file_path)
args = loaded_data['args']
h = loaded_data['h']
w = loaded_data['w']
c = loaded_data['c']
num_target = loaded_data['num_target']
num_server_client = loaded_data['num_server_client']
train_dataloader = loaded_data['train_dataloader']
test_dataloader = loaded_data['test_dataloader']
train_test_dataloader = loaded_data['train_test_dataloader']
public_dataloader = loaded_data['public_dataloader']
log = loaded_data['log']
num_target = loaded_data['num_target']
server_client = loaded_data['server_client']
neighbor_server = loaded_data['neighbor_server']
all_server = loaded_data['all_server']

# %% 模型初始化
if args.model_select == 1:
    model = CNN(h, w, c, num_target).to(args.device)
    client_model = list_same_term(args.num_all_client, model)
    server_model = list_same_term(args.num_all_server, model)
elif args.model_select == 2:
    model = LeNet5(h, w, c, num_target).to(args.device)
    client_model = list_same_term(args.num_all_client, model)
    server_model = list_same_term(args.num_all_server, model)
elif args.model_select == 3:
    model1 = CNN(h, w, c, num_target).to(args.device)
    model2 = LeNet5(h, w, c, num_target).to(args.device)
    model3 = MLP(h, w, c, 50, num_target).to(args.device)
    server_model = [model1, model2, model3]
    client_model1 = list_same_term(num_server_client, model1)
    client_model2 = list_same_term(num_server_client, model2)
    client_model3 = list_same_term(num_server_client, model3)
    client_model = [client_model1, client_model2, client_model3]
else:
    raise ValueError('model error.')

server_accuracy = list_same_term(args.num_all_server)
client_accuracy = list_same_term(args.num_all_client)
train_accuracy = list_same_term(args.num_all_client)
server_client_model = deepcopy(server_accuracy)
client_loss = deepcopy(client_accuracy)

# %% 模型训练
keys = ['server_model',
        'train_dataloader',
        'test_dataloader',
        'train_test_dataloader',
        'public_dataloader',
        'log',
        'client_model',
        'num_target',
        'neighbor',
        'client_idx',
        'client_accuracy',
        'client_loss',
        'train_accuracy']
values = [server_model,
          train_dataloader,
          test_dataloader,
          train_test_dataloader,
          public_dataloader,
          log,
          None,
          num_target,
          None,
          None,
          client_accuracy,
          client_loss,
          train_accuracy]
args_train = dict(zip(keys, values))

client_model_save = dict.fromkeys([i for i in range(args.num_client_commu)])
server_model_save = dict.fromkeys([i for i in range(args.num_server_commu)])

weight_server = list_same_term(args.num_all_server, 1/args.num_all_server)
weight_list = list_same_term(args.num_all_server, weight_server)

# %% 对每个服务器通讯幕进行循环
for epoch_server_commu in range(args.num_server_commu):
    log.info('-'*50)
    log.info('|epoch_server_commu: {}/{}'.format(epoch_server_commu,
             args.num_server_commu))
    # 所有边缘服务器分别协调其客户端进行多轮联邦学习
    for epoch_client_commu in range(args.num_client_commu):
        message = ' |epoch_client_commu: {}/{}'.format(
            epoch_client_commu, args.num_client_commu)
        log.info(message)
        # 所有边缘服务器分别协调其客户端进行联邦学习
        neighbor_model = []
        for i in range(epoch_client_commu):
            model, _ = train_model(model=client_model[rank],
                                dataloader=train_dataloader[rank],
                                device=args.device)
            client_model = comm.gather(model, root=0)
            if rank == 0:
                server_model = aggregate(client_model, [0.1, 0.2, 0.3, 0.4])
            A = comm.bcast(A, root=0)
        for server in all_server:
            # 每个服务器下单客户端分别训练
            message = f'  |server: {server}'
            log.info(message)
            args_train['client_idx'] = server_client[server]
            args_train['client_model'] = client_model
            if epoch_server_commu == 0:
                client_model = ServerTrain(args, args_train, 1).train
            else:
                for i in neighbor_server[server]:
                    neighbor_model.append(server_model[i])
                args_train['neighbor'] = neighbor_model
                client_model = ServerTrain(args, args_train, 4).train
            # torch.save(client_model, './test.pt')
            # break
            # 在单个服务器下客户端训练完成后更新该服务器下客户端的模型
            server_model_ = [
                client_model[client] for client in server_client[server]]
            # 聚合获得单个服务器模型并下发
            server_model[server] = aggregate(
                server_model_, weight_server)
            for client in server_client[server]:
                client_model[client] = deepcopy(server_model[server])
            # 评估单个服务器模型
            acc_server = eval_model(
                model=server_model[server],
                dataloader=test_dataloader,
                device=args.device)
            message = '|servers comunicated: {}, server aggregated: {}, acc_server {}: {:.3f}.'.format(
                epoch_server_commu, epoch_client_commu, server, acc_server)
            log.info(message)
            log.info('-'*50)
            server_accuracy[server].append(acc_server)
        # if epoch_client_commu == 1:
        #     torch.save(client_model, './client_model.pt')
        server_model_save[epoch_client_commu] = deepcopy(server_model)
        # break
        # client_model_save[epoch_client_commu] = deepcopy(client_model)
        if args.algorithm == 2:
            server_model = server_communicate(server_model, weight_list)
            for server in all_server:
                for client in server_client[server]:
                    client_model[client] = deepcopy(server_model[server])
    # torch.save(server_model, './server_model.pt')
    message = '{:^50}'.format('********  servers comunicates  ********')
    log.info(message)

# %% 保存
# save_data = {'args': args,
#              'server_model': server_model_save,
#              'server_acc': server_accuracy,
#              'client_model': client_model_save,
#              'client_acc': client_accuracy,
#              'train_acc': train_accuracy,
#              'client_loss': client_loss}
# save_file(args, save_data, log)
