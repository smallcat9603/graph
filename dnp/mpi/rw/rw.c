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

  // preparation
  int partner_rank = (rank+1)%2;
  int* rw = (int*) malloc(sizeof(int)*2);
  if (rank == 0) {
    rw[0] = 0;
    rw[1] = 1;
  }
  else {
    rw[0] = 3;
    rw[1] = 2;
  }
  
  // send and recv
  double start_time, end_time;
  start_time = MPI_Wtime();

  int* send = (int*) malloc(sizeof(int)*2); 
  int* recv = (int*) malloc(sizeof(int)*2);
  MPI_Status st[2];
  MPI_Request req[2];  

  MPI_Irecv(recv, 2, MPI_INT, partner_rank, 0, MPI_COMM_WORLD, req);
  
  memmove(send, rw, sizeof(int)*2);
  MPI_Isend(send, 2, MPI_INT, partner_rank, 0, MPI_COMM_WORLD, req+1);  

  MPI_Waitall(2, req, st); 

  // combination
  int* comb = (int*) malloc(sizeof(int)*4);
  memmove(comb, recv, sizeof(int)*2); 
  memmove(comb+2, &rw[1], sizeof(int)); 
  memmove(comb+3, &rw[0], sizeof(int)); 
  
  end_time = MPI_Wtime();

  // output
  printf("rank = %d, elapsed = %f: ", rank, end_time-start_time); 
  for(int i=0; i<4; i++) printf("%d --> ", comb[i]);
  printf("\n");                  

  // end
  MPI_Finalize();
}