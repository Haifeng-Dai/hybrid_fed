import torch
import numpy as np

from mpi4py import MPI
from torch.utils.data import DataLoader
from utils.data_util import get_dataset
from utils.model_util import *
from utils.train_util import *

_, data, _, _, _ = get_dataset()
dataloader = DataLoader(
    dataset=data, batch_size=100, shuffle=True)

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

A = CNN(28, 28, 1, 10)
for i in range(4):
    model, _, = train_model(
        model=A,
        dataloader=dataloader,
        device='cuda')
    print(i, rank, eval_model(model, dataloader, 'cuda'))
    # print(np.random.random(1))
    # print(i, 'comm.rank', comm.rank, 'a', a)
    # print(i, 'comm.rank', comm.rank, 'b', b)
    comm.barrier()
    A_list = comm.gather(model, root=0)
    if rank == 0:
        A = aggregate(A_list, [0.1, 0.2, 0.3, 0.4])
        # b = CNN(28, 28, 1, 10)
        # for key in A.state_dict():
        #     a = torch.eq(B[0].state_dict()[key], B[1].state_dict()[key])
        #     print(a)
    A = comm.bcast(A, root=0)
    print(i, rank, eval_model(A, dataloader, 'cuda'))
    comm.barrier()

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

# a = 1
# b = 0
# for i in range(4):
#     a += b
#     print(i, rank, b)
#     comm.barrier()
#     B = comm.gather(b, root=0)
#     # broadcast a generic object by using bcast
#     if rank == 0:
#         b += 1
#     comm.barrier()
#     b = comm.bcast(b, root=0)

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

# a = 1
# b = 0
# for i in range(4):
#     a += b
#     B = comm.gather(b, root=0)
#     # broadcast a generic object by using bcast
#     if rank == 0:
#         b += 1

#     b = comm.bcast(b, root=0)
#     print('rank %d has %s' % (rank, b))

# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

# for i in range(3):
#     a = 1
#     print('a', a)
#     comm.barrier()
#     print('b')
#     comm.barrier()


# import numpy as np
# from mpi4py import MPI


# comm = MPI.COMM_WORLD
# rank = comm.Get_rank()

# # ------------------------------------------------------------------------------
# # scatter a list of generic object by using scatter
# if rank == 0:
#     send_obj = [1.2, 'xxx', {'a': 1}, (2,)]
# else:
#     send_obj = None

# # each process receives one of the element of send_obj from rank 1
# #     rank 0   |   rank 1   |   rank 2   |   rank 3
# #  ------------+------------+------------+------------
# #      1.2     |   'xxx'    |  {'a': 1}  |   (2,)
# recv_obj = comm.scatter(send_obj, root=0)
# print('scatter: rank %d has %s' % (rank, recv_obj))
