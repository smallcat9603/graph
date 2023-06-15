//
// by smallcat 20230609
// 
// 
//

#include <stdio.h>
#include <igraph/igraph.h>

#include "file.h"

void map_nodes_in_edgelist(const char* file, const char* file_new, int* nnodes, int** node_map) {
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
  *node_map = (int*)malloc(sizeof(int)*num_new);
  for(int i=0; i<num_new; i++) (*node_map)[i] = nodes_new[i];

}

void read_edgelist(igraph_t* graph, const char* file, bool directed){
  FILE* fp;
  fp = fopen(file, "r");
  if (fp == NULL) {
      printf("there is something wrong with opening %s\n", file);
      exit(0);
  }
  igraph_read_graph_edgelist(graph, fp, 0, directed);
  fclose(fp);
}
