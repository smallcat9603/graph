//
// by smallcat 20230609
// 
// 
//

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <string.h>
#include <unistd.h>

int main(int argc, char** argv) {

  MPI_Init(NULL, NULL);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  double start_time, end_time, last_received_time;
  start_time = MPI_Wtime();
  last_received_time = MPI_Wtime();

  int* send = (int*) malloc(sizeof(int)*2);
  send[0] = rank;
  send[1] = rank + 1;
  int partner_rank = (rank+1)%2;
  int count = 2;

  MPI_Status status;
  int flag = 0;

  while (count > 0){
  MPI_Request req;
  MPI_Isend(send, 2, MPI_INT, partner_rank, 0, MPI_COMM_WORLD, &req);  
  count--;
  }

  while (MPI_Wtime() - last_received_time < 3) {

    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
    if (flag) {
      int received_count;
      MPI_Get_count(&status, MPI_INT, &received_count);
      printf("rank = %d, received_count=%d\n", rank, received_count);
      last_received_time = MPI_Wtime();
      MPI_Request req_recv;
      MPI_Status st;
      int* recv = (int*) malloc(sizeof(int)*2);
      MPI_Irecv(recv, received_count, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &req_recv);
      printf("rank=%d, recv[0]=%d, recv[1]=%d\n", rank, recv[0], recv[1]);
    } 

  }

  end_time = MPI_Wtime();

  printf("rank = %d, elapsed = %f\n", rank, end_time-start_time);              

  MPI_Finalize();
}