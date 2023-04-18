#
# A toy MPI program
# Graph: 0-1-2-3
# 0,1 are hosted by host0, 2,3 are hosted by host1 
# 

from mpi4py import MPI
import time

# initialize
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

if size != 2:
  print("World size must be two!")
  exit(1)

# preparation
partner_rank = (rank+1)%2
rw = []
if rank == 0:
  rw.append(0)
  rw.append(1)
else:
  rw.append(3)
  rw.append(2)  

# send and recv
start_time = time.time()

req = comm.irecv(source=partner_rank, tag=0)

send = rw.copy()
comm.isend(send, dest=partner_rank, tag=0)

recv = req.wait()

# combination
comb = recv.copy()
comb.append(rw[1])
comb.append(rw[0])

end_time = time.time()

# output
print("rank = {0}, elapsed = {1}: {2}".format(rank, end_time-start_time, comb)) 