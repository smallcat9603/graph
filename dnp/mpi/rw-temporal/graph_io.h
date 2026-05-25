/*
 * graph_io.h -- partition (subgraph) loading and result log writing.
 *
 * Each rank owns a partition. The on-disk edgelist uses sparse global ids;
 * we densify them to local ids [0..nnodes-1] on load and keep both
 * directions of the mapping:
 *
 *     l2g[local_id]                  = global_id     (array)
 *     intmap_get(&g2l, global_id)    = local_id      (hash map)
 */
#ifndef GRAPH_IO_H
#define GRAPH_IO_H

#include <stdbool.h>
#include <igraph/igraph.h>

#include "intmap.h"

typedef struct {
    igraph_t graph;
    int*     l2g;      /* size = nnodes */
    int      nnodes;
    intmap_t g2l;
} partition_t;

void partition_init(partition_t* p);
void partition_free(partition_t* p);

/* Read an edgelist "src dst" (whitespace separated, global ids) and build
 * the igraph + l2g + g2l. Builds igraph in-memory, no temp files.
 * Returns 0 on success, -1 on file-open failure. */
int  partition_load_edgelist(partition_t* p, const char* path, bool directed);

/* Abort the program if the loaded graph is not weakly connected. */
void partition_assert_connected(const partition_t* p);

/* Write `nwalkers` rows of `walker_len` ints each (space separated, one
 * row per line) to `path`. Returns 0 on success, -1 on open failure. */
int log_write(const char* path, const int* paths, int nwalkers, int walker_len);

#endif /* GRAPH_IO_H */
