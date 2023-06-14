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

int* map_nodes_in_edgelist(const char* file, const char* file_new, int* nnodes) {
  FILE *fp;

  //read file
  fp = fopen(file, "r");
  if (fp == NULL) {
      printf("there is something wrong with opening %s\n", file);
      exit(0);
  }
  int src, dst;
  int num = 0;
  int* nodes = NULL;
  while (fscanf(fp, "%d %d", &src, &dst) == 2) {
    num = num + 2;
    nodes = (int*)realloc(nodes, sizeof(int)*num);
    nodes[num-2] = src;
    nodes[num-1] = dst;
  }
  fclose(fp);

  int num_new = num;
  int nodes_new[num_new];
  for (int i = 0; i < num_new; i++) nodes_new[i] = nodes[i];

  //remove duplicates in nodes
  for(int i=0; i<num_new-1; i++)	
	{ 
    for(int j=i+1; j<num_new; j++){
      if(nodes_new[i] == nodes_new[j])
      { 
        for(int k=j; k<num_new-1; k++) nodes_new[k] = nodes_new[k+1];
        num_new--;
        j--;
      }
    }
  }

  //write to file_new
  fp = fopen(file_new, "w");
  for(int i = 0; i < num; i=i+2){
    int src = nodes[i], dst = nodes[i+1];
    int src_new, dst_new;
    int cnt = 0;
    for(int j = 0; j < num_new; j++){
      if(src == nodes_new[j]){
        src_new = j;
        cnt++;
      }
      if(dst == nodes_new[j]){
        dst_new = j;
        cnt++;
      }
      if(cnt == 2) break;
    }
    fprintf(fp, "%d %d\n", src_new, dst_new);
  }
  fclose(fp);

  *nnodes = num_new;
  return nodes_new;
}

void read_edgelist(igraph_t* graph, const char* file, bool directed){
  FILE* fp;
  fp = fopen(file, "r");
  if (file == NULL) {
      printf("there is something wrong with opening %s\n", file);
      exit(0);
  }
  igraph_read_graph_edgelist(graph, fp, 0, directed);
  fclose(fp);
}


int jump(){

}

int walk(){

}

int main(int argc, char** argv) {
  char graphbase[100] = "facebook";
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

  int walker_id_start = rank * nwalkers;
  printf("rank = %d/%d, nwalkers = %d (%d-%d), nsteps = %d\n", rank, size, nwalkers, walker_id_start, walker_id_start+nwalkers-1, nsteps); 

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

  char file[100];
  char file_new[100];
  sprintf(file, "%s.txt", graphbase);
  if(size > 1) sprintf(file, "%s.sub%d.txt", graphbase, rank);
  sprintf(file_new, "%s.sub%d.x.txt", graphbase, rank);
  int nnodes;
  int* node_map = map_nodes_in_edgelist(file, file_new, &nnodes);
  igraph_t graph;
  read_edgelist(&graph, file_new, false); //true=directed, false=undirected
  printf("generated graph from file %s\n", file_new);
  printf("nnodes = %ld, nedges = %ld\n", igraph_vcount(&graph), igraph_ecount(&graph));

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