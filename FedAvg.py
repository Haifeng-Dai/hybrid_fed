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
device='cuda'
comm = MPI.COMM_WORLD
rank = comm.Get_rank()

A = CNN(28, 28, 1, 10)
for i in range(4):
    model, _ = train_model(
        model=A.to(device),
        dataloader=dataloader,
        device=device)
    print('train', i, rank, eval_model(model, dataloader, device))
    A_list = comm.gather(model, root=0)
    print(type(A_list))
    if rank == 0:
        A = aggregate(A_list, [0.1, 0.2, 0.3, 0.4])
        print('bcast', i, rank, eval_model(A, dataloader, device))
    A = comm.bcast(A, root=0)