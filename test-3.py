from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

a = 0
for i in range(4):
    a = a + rank
    A = comm.gather(a, root=0)
    if rank == 0:
        a = sum(A)
        print('a:', a, 'A:', A)
    a = comm.bcast(a, root=0)
