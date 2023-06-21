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
#include <igraph/igraph.h>

#include "algo.h"

void nexthop_roulette(igraph_t* graph, rt* dict, int rt_size, int* node_map, int cur_local, int cur_global, int* next_local_node, int* next_global_node, int* next_global_proc){
    igraph_vector_int_t neighbors_in;
    igraph_vector_int_init(&neighbors_in, 0);
    igraph_neighbors(graph, &neighbors_in, (igraph_integer_t)cur_local, IGRAPH_ALL);
    int nneighbors_in = igraph_vector_int_size(&neighbors_in);
    int* neighbors_out = NULL;
    int nneighbors_out = 0;
    int idx = get_rt(dict, rt_size, cur_global);
    if(idx != -1){
        neighbors_out = dict[idx].dst_proc;
        nneighbors_out = dict[idx].num/2;
    }
    int nneighbors = nneighbors_in + nneighbors_out;
    int next_idx = rand()%nneighbors; 
    *next_local_node = -1, *next_global_node = -1, *next_global_proc = -1;
    if(next_idx < nneighbors_in){ // next node is inside 
        *next_local_node = (int)VECTOR(neighbors_in)[next_idx];
        *next_global_node = node_map[*next_local_node];
    }
    else{ // next node is outside
        int next_global_node_idx = (next_idx-nneighbors_in)*2;
        *next_global_node = neighbors_out[next_global_node_idx];
        *next_global_proc = neighbors_out[next_global_node_idx+1];
    }  
}

void walk(igraph_t* graph, rt* dict, int rt_size, int** walker, int* len, int* node_map, int nnodes, int** paths, int *npaths, int nsteps){
    int next_local_node = -1, next_global_node = -1, next_global_proc = -1;
    int id = (*walker)[0];
    int LEN = RSV_INTS + nsteps;

    while(next_global_proc == -1 && *len < LEN){ //walk inside
        int cur_local = -1, cur_global = -1;
        if(*len == RSV_INTS){ //starting point of walker
            printf("Walker%d gets started to walk.\n", id);
            cur_local = rand() % nnodes;
            cur_global = node_map[cur_local];
            if(cur_local == -1 || cur_global == -1){
                printf("there is something wrong with idx local --> global\n");
                exit(0);
            } 
            (*len)++;
            *walker = (int*)realloc(*walker, sizeof(int)*(*len));
            (*walker)[*len-1] = cur_global;
        }
        else{
            cur_global = (*walker)[*len-1];
            for(int i = 0; i < nnodes; i++){
                if(node_map[i] == cur_global){
                    cur_local = i;
                    break;
                }
            }
        }
        if(cur_local == -1 || cur_global == -1){
            printf("there is something wrong with idx global --> local\n");
            exit(0);
        } 
        nexthop_roulette(graph, dict, rt_size, node_map, cur_local, cur_global, &next_local_node, &next_global_node, &next_global_proc);
        (*len)++;
        *walker = (int*)realloc(*walker, sizeof(int)*(*len));
        (*walker)[*len-1] = next_global_node;      
    }

    if(*len >= LEN){
        printf("Finished. Walker%d stopped.\n", id);
        (*walker)[2] = (int)time(NULL);
        // for(int i=0; i<*len; i++){
        //     printf("%d ", (*walker)[i]);
        // }
        // printf("\n");
        (*npaths)++;
        *paths = (int*)realloc(*paths, sizeof(int)*(*len)*(*npaths));
        memmove(*paths+(*len)*(*npaths-1), *walker, sizeof(int)*(*len));
        free(*walker);
    }
    else if(next_local_node == -1){ //walk outside
        (*walker)[3] += 1;       
        // MPI_Request req;
        // MPI_Isend(*walker, *len, MPI_INT, next_global_proc, 0, MPI_COMM_WORLD, &req);  
        // use block communication instead of non-block one, otherwise recv will have chance to receive multiple isends at once
        MPI_Send(*walker, *len, MPI_INT, next_global_proc, 0, MPI_COMM_WORLD);
        free(*walker);
    }
    else{
        printf("Something is wrong for Walker%d\n", id);
        exit(0);       
    }

}

void gen_walker(int** walker, int id, int len){
    *walker = (int*) malloc(sizeof(int)*len);
    (*walker)[0] = id; 
    (*walker)[1] = (int)time(NULL); //start
    (*walker)[2] = (int)time(NULL); //end
    (*walker)[3] = 0; //go_out
}

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
