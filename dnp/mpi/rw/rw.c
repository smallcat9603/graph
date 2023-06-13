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
#include <igraph/igraph.h>

int jump(){

}

int walk(){

}

int main(int argc, char** argv) {

  igraph_integer_t num_vertices = 1000;
  igraph_integer_t num_edges = 1000;
  igraph_real_t diameter;
  igraph_t graph;

  igraph_rng_seed(igraph_rng_default(), 42);

  igraph_erdos_renyi_game(&graph, IGRAPH_ERDOS_RENYI_GNM,
                          num_vertices, num_edges,
                          IGRAPH_UNDIRECTED, IGRAPH_NO_LOOPS);

  igraph_diameter(&graph, &diameter, NULL, NULL, NULL, NULL, IGRAPH_UNDIRECTED, /* unconn= */ true);
  printf("Diameter of a random graph with average degree %g: %g\n",
          2.0 * igraph_ecount(&graph) / igraph_vcount(&graph),
          (double) diameter);

  igraph_destroy(&graph);

	int nwalkers = 1;
  int nsteps = 80;
	if(argc > 1) nwalkers = atoi(argv[1]);
  if(argc > 2) nsteps = atoi(argv[2]); 

  MPI_Init(NULL, NULL);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int walker_id_start = rank * nwalkers;
  printf("rank = %d/%d, nwalkers = %d (%d-%d), nsteps = %d\n", rank, size, nwalkers, walker_id_start, walker_id_start+nwalkers-1, nsteps); 

  double start, end, tail=3.0;
  start = MPI_Wtime();
  end = MPI_Wtime();

  int partner_rank = (rank+1)%2;

  for(int id=walker_id_start; id<walker_id_start+nwalkers; id++){
    int* walker = (int*) malloc(sizeof(int)*1);
    walker[0] = rank;
    MPI_Request req;
    MPI_Isend(walker, 1, MPI_INT, partner_rank, id, MPI_COMM_WORLD, &req);  
    free(walker);
  }

  MPI_Status status;
  int flag = 0;

  while(MPI_Wtime() - end < tail) {

    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
    if(flag) {
      int count;
      MPI_Get_count(&status, MPI_INT, &count);
      // printf("rank = %d, count=%d\n", rank, count);
      int* recv = (int*) malloc(sizeof(int)*count);
      MPI_Status st;
      MPI_Recv(recv, count, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &st);
      end = MPI_Wtime();
      if(count < nsteps){
        int* send = (int*)malloc(sizeof(int)*(count+1));
        memmove(send, recv, sizeof(int)*count);
        send[count] = send[count-1] + 1;
        MPI_Request req;
        MPI_Isend(send, count+1, MPI_INT, partner_rank, st.MPI_TAG, MPI_COMM_WORLD, &req);
        free(send);
      }
      else{
        printf("rank = %d, walker = %d:\n", rank, st.MPI_TAG);
        for(int i=0; i<count; i++){
          printf(" %d", recv[i]);
        }
        printf("\n");
      }
      free(recv);
    } 
  }

  printf("rank = %d, elapsed = %f\n", rank, end-start);              

  MPI_Finalize();
}