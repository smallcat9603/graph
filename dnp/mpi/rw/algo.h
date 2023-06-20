//
// by smallcat 20230609
// 
// 
//

#ifndef ALGO_H
#define ALGO_H

#include "rt.h"

#define RSV_INTS 4

void nexthop_roulette(igraph_t* graph, rt* dict, int rt_size, int* node_map, int cur_local, int cur_global, int* next_local_node, int* next_global_node, int* next_global_proc);
void walk(igraph_t* graph, rt* dict, int rt_size, int** walker, int* len, int* node_map, int nnodes, int** paths, int *npaths, int nsteps);
void gen_walker(int** walker, int id, int len);

#endif
