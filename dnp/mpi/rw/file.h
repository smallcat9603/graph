//
// by smallcat 20230609
// 
// 
//

#ifndef FILE_H
#define FILE_H

void map_nodes_in_edgelist(const char* file, const char* file_new, int* nnodes, int** node_map);
void read_edgelist(igraph_t* graph, const char* file, bool directed);

#endif
