/********************************************************************
 This program is for mpi random walk.
 ----------------------------------------------
 Email : huyao0107@gmail.com
 ---------------------------------------------------------------
********************************************************************/

#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <igraph/igraph.h>

#include "algo.h"
#include "file.h"
#include "rt.h"

void check_graph(igraph_t* graph){
  igraph_bool_t connected;
  igraph_connectedness_t mode = IGRAPH_WEAK;
  igraph_is_connected(graph, &connected, mode);
  if(connected){
    printf("Graph is connected\n");
    printf("nnodes = %ld, nedges = %ld\n", igraph_vcount(graph), igraph_ecount(graph));
  }
  else{
    printf("Graph is not connected\n");
    igraph_vector_int_t membership;
    igraph_vector_int_t csize;
    igraph_integer_t no;
    igraph_connectedness_t mode = IGRAPH_WEAK;
    igraph_connected_components(graph, &membership, &csize, &no, mode);
    printf("Graph is composed of %ld components", no);
    exit(0);
  }
}

int main(int argc, char** argv) {
  char graphbase[192] = "facebook";
	int nwalkers = 1;
  int nsteps = 80;
  if(argc > 1) strcpy(graphbase, argv[1]);
	if(argc > 2) nwalkers = atoi(argv[2]);
  if(argc > 3) nsteps = atoi(argv[3]); 

  MPI_Init(NULL, NULL);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int id_start = rank * nwalkers;
  printf("rank = %d/%d, nwalkers = %d (%d-%d), nsteps = %d\n", rank, size, nwalkers, id_start, id_start+nwalkers-1, nsteps); 

  if(size < 2){
    if(strcmp(graphbase, "facebook") == 0) strcpy(graphbase, "../../pyro/rw/data/facebook_combined_undirected_connected");
    else if(strcmp(graphbase, "git") == 0) strcpy(graphbase, "../../pyro/rw/data/musae_git_edges_undirected.connected");
    else if(strcmp(graphbase, "twitch") == 0) strcpy(graphbase, "../../pyro/rw/data/large_twitch_edges_undirected.connected");
    else if(strcmp(graphbase, "livejournal") == 0) strcpy(graphbase, "../../pyro/rw/data/soc-LiveJournal1_directed.undirected.connected");
  }
  else{
    if(strcmp(graphbase, "facebook") == 0) sprintf(graphbase, "../../pyro/rw/data/%d/facebook_combined_undirected_connected", size);
    else if(strcmp(graphbase, "git") == 0) sprintf(graphbase, "../../pyro/rw/data/%d/musae_git_edges_undirected.connected", size);
    else if(strcmp(graphbase, "twitch") == 0) sprintf(graphbase, "../../pyro/rw/data/%d/large_twitch_edges_undirected.connected", size);
    else if(strcmp(graphbase, "livejournal") == 0) sprintf(graphbase, "../../pyro/rw/data/%d/soc-LiveJournal1_directed.undirected.connected", size);
  }

  char file[256];
  char file_new[256];
  sprintf(file, "%s.txt", graphbase);
  if(size > 1) sprintf(file, "%s.sub%d.txt", graphbase, rank);
  sprintf(file_new, "%s.sub%d.x.txt", graphbase, rank);
  int nnodes;
  int* node_map;
  map_nodes_in_edgelist(file, file_new, &nnodes, &node_map);
  igraph_t graph;
  read_edgelist(&graph, file_new, false); //true=directed, false=undirected
  printf("generated graph from file %s\n", file_new);

  // check graph
  check_graph(&graph);

  //read route table
  char file_rt[256];
  rt* dict = NULL;
  int rt_size = 0;
  if(size > 1){
    sprintf(file_rt, "%s.rt%d.txt", graphbase, rank);
    read_rt(file_rt, &dict, &rt_size);
    printf("read rt file %s\n", file_rt);
  }

  double start, end, tail=3.0;
  start = MPI_Wtime();
  end = MPI_Wtime();

  int* paths = NULL;
  int npaths = 0;

  // start walkers
  for(int id=id_start; id<id_start+nwalkers; id++){
    int* walker;
    int len = RSV_INTS;
    gen_walker(&walker, id, len);
    walk(&graph, dict, rt_size, &walker, &len, node_map, nnodes, &paths, &npaths, nsteps);
  }

  MPI_Status status;
  int flag = 0;

  while(MPI_Wtime() - end < tail) {

    MPI_Iprobe(MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &flag, &status);
    if(flag) {
      int count;
      MPI_Get_count(&status, MPI_INT, &count);
      int* recv = (int*) malloc(sizeof(int)*count);
      MPI_Recv(recv, count, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
      end = MPI_Wtime();
      walk(&graph, dict, rt_size, &recv, &count, node_map, nnodes, &paths, &npaths, nsteps);
    }
  }

  printf("rank = %d, elapsed = %f\n", rank, end-start);              

  MPI_Finalize();
}