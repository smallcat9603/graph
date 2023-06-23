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

  int total_nwalkers = nwalkers * size;
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
  sprintf(file_new, "%s.x.txt", graphbase);
  if(size > 1){
    sprintf(file, "%s.sub%d.txt", graphbase, rank);
    sprintf(file_new, "%s.sub%d.x.txt", graphbase, rank);
  }
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

  int* paths = NULL;
  int npaths = 0;
  int sum_npaths = 0;

  srand(time(NULL));

  double start, end;
  start = MPI_Wtime();

  // start walkers
  for(int id=id_start; id<id_start+nwalkers; id++){
    int* walker = NULL;
    int len = RSV_INTS;
    gen_walker(&walker, id, len);
    walk(&graph, dict, rt_size, &walker, &len, node_map, nnodes, &paths, &npaths, nsteps);
  }

  //recv and walk
  MPI_Status status;
  int flag = 0;
  while(sum_npaths < total_nwalkers) {
    MPI_Iprobe(MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &flag, &status);
    if(flag) {
      int count;
      MPI_Get_count(&status, MPI_INT, &count);
      int* recv = (int*) malloc(sizeof(int)*count);
      MPI_Recv(recv, count, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
      walk(&graph, dict, rt_size, &recv, &count, node_map, nnodes, &paths, &npaths, nsteps);
    }
    // MPI_Reduce(&npaths, &sum_npaths, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    // MPI_Bcast(&sum_npaths, 1, MPI_INT, 0, MPI_COMM_WORLD);
    // MPI_Reduce + MPI_Bcast
    MPI_Allreduce(&npaths, &sum_npaths, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
  }

  end = MPI_Wtime();

  //delete temporal .x.txt
  if (remove(file_new) == 0) {
    printf("deleted %s\n", file_new);
  } else {
    printf("%s deletion failed\n", file_new);
  }

  //gather results from each process
  int buf_npaths[size];
  MPI_Gather(&npaths, 1, MPI_INT, buf_npaths, 1, MPI_INT, 0, MPI_COMM_WORLD);

  int displacements[size];
  displacements[0] = 0;
  for(int i = 1; i < size; i++){
    displacements[i] = 0;
    for(int j = 0; j < i; j++){
      displacements[i] += buf_npaths[j];
    } 
  }

  int LEN = RSV_INTS + nsteps;
  for(int i = 0; i < size; i++){
    buf_npaths[i] *= LEN;
    displacements[i] *= LEN;
  }
  int buf_size = sum_npaths * LEN;
  int buf_paths[buf_size];
  MPI_Gatherv(paths, npaths*LEN, MPI_INT, buf_paths, buf_npaths, displacements, MPI_INT, 0, MPI_COMM_WORLD);
  
  //print result
  if (rank == 0) {
    char log[256];
    sprintf(log, "log/%d_%s_w%d_s%d_p%d.txt", (int)end, argv[1], nwalkers, nsteps, size);
    FILE* fp = fopen(log, "w");
    for(int i = 0; i < buf_size; i++)
    {
      fprintf(fp, "%d ", buf_paths[i]);
      if((i+1)%LEN == 0) fprintf(fp, "\n");
    }
    fclose(fp);
    printf("%s generated.\n", log); 
    printf("rank = %d, elapsed = %f\n", rank, end-start); 
  }             

  MPI_Finalize();
}