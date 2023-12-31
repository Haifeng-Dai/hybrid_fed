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

log = get_logger(log_path)

for epoch_server_commu in range(num_server_client):
    log.info('-'*50)
    log.info('|epoch_server_commu: {}/{}'. format(epoch_server_commu +
             1, args.num_server_commu))
    # 所有边缘服务器分别协调其客户端进行多轮联邦学习
    for epoch_client_commu in range(args.num_client_commu):
        message = ' |epoch_client_commu: {}/{}'.format(
            epoch_client_commu + 1, args.num_client_commu)
        log.info(message)
        # 客户端训练
        loss_ = []
        acc_ = []
        if args.algorithm == 0:  # FedAvg without public data
            for epoch_client_train in range(args.num_client_train):
                model, loss__ = train_model(model=client_model[rank],
                                            dataloader=train_dataloader[rank],
                                            device=args.device)
                loss_.extend(loss__)
                acc__ = eval_model(model=model,
                                   dataloader=test_dataloader,
                                   device=args.device)
                acc_.append(acc__)
                message = '  |epoch_client_train: {}/{}, loss: {}, acc: {}'.format(
                    epoch_client_train + 1, args.num_client_train, loss__[-1], acc__)
                log.info(message)
        elif args.algorithm == 1:  # FedAvg with public data
            for epoch_client_train in range(args.num_client_train):
                model, loss__ = train_model(model=client_model[rank],
                                            dataloader=train_dataloader[rank],
                                            device=args.device)
                loss_.extend(loss__)
                acc__ = eval_model(model=model,
                                   dataloader=test_dataloader,
                                   device=args.device)
                acc_.append(acc__)
                message = '  |epoch_client_train: {}/{}, loss: {}, acc: {}'.format(
                    epoch_client_train + 1, args.num_client_train, loss__[-1], acc__)
                log.info(message)
            for epoch_public_train in range(args.num_public_train):
                model, loss__ = train_model(model=model,
                                            dataloader=public_dataloader,
                                            device=args.device)
                loss_.extend(loss__)
                acc__ = eval_model(model=model,
                                   dataloader=test_dataloader,
                                   device=args.device)
                acc_.append(acc__)
                message = '  |epoch_client_train: {}/{}, loss: {}, acc: {}'.format(
                    epoch_client_train + 1, args.num_client_train, loss__[-1], acc__)
                log.info(message)
        elif args.algorithm == 2:  # FedAvg with single distill
            for epoch_client_train in range(args.num_client_train):
                model, loss__ = train_model(model=client_model[rank],
                                            dataloader=train_dataloader[rank],
                                            device=args.device)
                loss_.extend(loss__)
                acc__ = eval_model(model=model,
                                   dataloader=test_dataloader,
                                   device=args.device)
                acc_.append(acc__)
                message = '  |epoch_client_train: {}/{}, loss: {}, acc: {}'.format(
                    epoch_client_train + 1, args.num_client_train, loss__[-1], acc__)
                # [[1], [2], [0]] / 0, [1]
                for server, servers in enumerate(neighbor_server):
                    if rank in server_client[server]:
                        neighbors = deepcopy(servers)
                        break
                for server in neighbors:
                    for epoch_public_train in range(args.num_public_train):
                        model, loss__ = train_model_disti_single(
                            model=model,
                            teacher_model=server_model[server],
                            dataloader=public_dataloader,
                            alpha=args.alpha,
                            T=args.T,
                            device=args.device)
                        loss_.extend(loss__)
                        acc__ = eval_model(model=model,
                                           dataloader=test_dataloader,
                                           device=args.device)
                        acc_.append(acc__)
                        message = '  |epoch_client_train: {}/{}, loss: {}, acc: {}'.format(
                            epoch_client_train + 1, args.num_client_train, loss__[-1], acc__)
                        log.info(message)
        elif args.algorithm == 3:  # FedAvg with weight distill
            for epoch_client_train in range(args.num_client_train):
                model, loss__ = train_model(model=client_model[rank],
                                            dataloader=train_dataloader[rank],
                                            device=args.device)
                loss_.extend(loss__)
                acc__ = eval_model(model=model,
                                   dataloader=test_dataloader,
                                   device=args.device)
                acc_.append(acc__)
                message = '  |epoch_client_train: {}/{}, loss: {}, acc: {}'.format(
                    epoch_client_train + 1, args.num_client_train, loss__[-1], acc__)
                log.info(message)
            for server, servers in enumerate(neighbor_server):
                if rank in server_client[server]:
                    neighbors = deepcopy(servers)
                    break
            for epoch_public_train in range(args.num_public_train):
                model, loss__ = train_model_disti_weighted(
                    model=model,
                    weight=[1/3, 1/3, 1/3],
                    alpha=args.alpha,
                    T=args.T,
                    dataloader=public_dataloader,
                    num_target=num_target,
                    neighbor=neighbors,
                    device=args.device)
                loss_.extend(loss__)
                acc__ = eval_model(model=model,
                                   dataloader=test_dataloader,
                                   device=args.device)
                acc_.append(acc__)
                message = '  |epoch_client_train: {}/{}, loss: {}, acc: {}'.format(
                    epoch_client_train + 1, args.num_client_train, loss__[-1], acc__)
                log.info(message)
        elif args.algorithm == 4:  # FedAvg with servers communication
            for epoch_client_train in range(args.num_client_train):
                model, loss__ = train_model(model=client_model[rank],
                                            dataloader=train_dataloader[rank],
                                            device=args.device)
                loss_.extend(loss__)
                acc__ = eval_model(model=model,
                                   dataloader=test_dataloader,
                                   device=args.device)
                acc_.append(acc__)
                message = '  |epoch_client_train: {}/{}, loss: {}, acc: {}'.format(
                    epoch_client_train + 1, args.num_client_train, loss__[-1], acc__)
                log.info(message)
        else:
            raise ValueError('algorithm error.')
        # gather model
        client_model = comm.gather(model, root=0)
        # gather loss and accuracy
        loss = comm.gather(loss_, root=0)
        acc = comm.gather(acc_, root=0)
        # 聚合
        if rank == 0:
            # aggregate loss and accuracy
            for client in all_client:
                client_loss[client].extend(loss[client])
                client_accuracy[client].extend(acc[client])
            # server weighted aggregate
            # if args.algorithm == 4:
            for server, clients in enumerate(server_client):
                clients_model = [client_model[i] for i in clients]
                server_model[server] = aggregate(
                    clients_model, [0.2, 0.3, 0.5])
                server_accuracy[server].append(eval_model(model=server_model[server],
                                                          dataloader=test_dataloader,
                                                          device=args.device))
                for client in clients:
                    client_model[client] = deepcopy(server_model[server])
        client_model = comm.bcast(client_model, root=0)
    # server communication
    if args.algorithm == 4:
        if rank == 0:
            weight_list = list_same_term(3, [1/3, 1/3, 1/3])
            server_model = server_communicate(server_model, weight_list)
            for server in all_server:
                for client in server_client[server]:
                    client_model[client] = deepcopy(server_model[server])
            client_model = comm.bcast(client_model, root=0)
    if args.algorithm == 2 or args.algorithm == 3:
        server_model = comm.bcast(server_model, root=0)
