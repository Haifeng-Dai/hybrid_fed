from mpi4py import MPI

com = MPI.COMM_WORLD

a = [1, 2, 3, 4]
b = com.rank
print(a, b)
