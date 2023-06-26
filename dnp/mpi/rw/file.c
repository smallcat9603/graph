//
// by smallcat 20230609
// 
// 
//

#include <stdio.h>
#include <igraph/igraph.h>

#include "file.h"

void map_nodes_in_edgelist(const char* file, const char* file_new, int* nnodes, int** node_map) {
  int src, dst;
  
  FILE *fp;
  //read file
  fp = fopen(file, "r");
  if (fp == NULL) {
      printf("there is something wrong with opening %s\n", file);
      exit(0);
  }
  while (fscanf(fp, "%d %d", &src, &dst) == 2) {
    int i;
    if(*nnodes > 0){
      for(i = 0; i < *nnodes; i++){
        if(src == (*node_map)[i]){
          break;
        }
      }
      if(i == *nnodes){
        (*nnodes)++;
        *node_map = (int*)realloc(*node_map, sizeof(int)*(*nnodes));
        (*node_map)[(*nnodes)-1] = src;
      }
      for(i = 0; i < *nnodes; i++){
        if(dst == (*node_map)[i]){
          break;
        }
      }
      if(i == *nnodes){
        (*nnodes)++;
        *node_map = (int*)realloc(*node_map, sizeof(int)*(*nnodes));
        (*node_map)[(*nnodes)-1] = dst;
      }      
    }
    else{
      (*nnodes)++;
      *node_map = (int*)malloc(sizeof(int)*(*nnodes));
      (*node_map)[(*nnodes)-1] = src;
    }
  }
  fclose(fp);

  //write to file_new
  FILE *fp_;
  FILE *fp_new;
  fp_ = fopen(file, "r");
  fp_new = fopen(file_new, "w");
  while (fscanf(fp_, "%d %d", &src, &dst) == 2) {
    int src_new, dst_new;
    int cnt = 0;
    for(int i = 0; i < *nnodes; i++){
      if(src == (*node_map)[i]){
        src_new = i;
        cnt++;
      }
      if(dst == (*node_map)[i]){
        dst_new = i;
        cnt++;
      }
      if(cnt == 2) break;
    }
    fprintf(fp_new, "%d %d\n", src_new, dst_new);
  }
  fclose(fp_);
  fclose(fp_new);
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
