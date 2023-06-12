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

int main(int argc, char** argv) {

  // Initialize the MPI environment
  MPI_Init(NULL, NULL);
  // Find out rank, size
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // We are assuming at least 2 processes for this task
  if (size != 2) {
    fprintf(stderr, "World size must be two for %s\n", argv[0]);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  MPI_Request req;
  MPI_Status st;
  int received_flag = 0;

  double start_time, end_time;
  start_time = MPI_Wtime();

  while (!received_flag) {

    int* recv = (int*) malloc(sizeof(int)*2);
    MPI_Irecv(recv, 2, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &req);


    MPI_Test(&req, &received_flag, &st);
    if (received_flag) {
      int received_count;
      MPI_Get_count(&st, MPI_INT, &received_count);
      printf("received_count=%d\n", received_count);
    } else {
      int* send = (int*) malloc(sizeof(int)*2);
      send[0] = rank;
      send[1] = rank + 1;
      int partner_rank = (rank+1)%2;
      MPI_Isend(send, 2, MPI_INT, partner_rank, 0, MPI_COMM_WORLD, &req);  
      MPI_Test(&req, &received_flag, &st);
    }
  }

  end_time = MPI_Wtime();

  printf("rank = %d, elapsed = %f\n", rank, end_time-start_time);              

  MPI_Finalize();
}