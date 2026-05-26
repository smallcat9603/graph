/*
 * graph_io.h -- partition (subgraph) loading and result log writing.
 *
 * Each rank owns a partition stored as a per-local-node Time-sorted
 * Adjacency List (TAL). For local node `local_id`, `tals[local_id]` is a
 * `tal_t` whose `edges` array is sorted by `t` ascending. Looking up
 * "first edge with t > t_cur" is then an O(log deg) binary search.
 *
 * Sparse global node ids are densified to local ids [0..nnodes-1] on load:
 *
 *     l2g[local_id]                  = global_id     (array)
 *     intmap_get(&g2l, global_id)    = local_id      (hash map)
 *
 * The input edgelist may be 2-column (`src dst`, timestamps synthesised
 * via synth_timestamp) or 3-column (`src dst t`, timestamps used as-is).
 * Each line is parsed independently, so a file may even mix the two,
 * though for clarity it should not.
 */
#ifndef GRAPH_IO_H
#define GRAPH_IO_H

#include <stdbool.h>

#include "intmap.h"

/* One entry in a node's time-sorted adjacency list. */
typedef struct {
    int neighbor_local;  /* local id of the neighbour within this partition */
    int t;               /* edge timestamp */
} tal_edge_t;

/* Time-sorted adjacency list for one local node. */
typedef struct {
    tal_edge_t* edges;   /* size `size`; sorted by `t` ascending */
    int         size;
} tal_t;

typedef struct {
    tal_t*   tals;    /* size = nnodes; one TAL per local node */
    int*     l2g;     /* size = nnodes */
    int      nnodes;
    intmap_t g2l;
    int      t_min;   /* smallest edge timestamp seen during load */
    int      t_max;   /* largest  edge timestamp seen during load */
} partition_t;

void partition_init(partition_t* p);
void partition_free(partition_t* p);

/* Read an edgelist file ("src dst" or "src dst t" per line, global ids)
 * and build the densified per-node TAL. Returns 0 on success, -1 on
 * file-open failure. */
int  partition_load_edgelist(partition_t* p, const char* path);

/* Find the index of the first edge in `tal` with t > t_cur, or tal->size
 * if no such edge exists. Branchless, cache-friendly binary search. */
static inline int tal_upper_bound(const tal_t* tal, int t_cur) {
    int lo = 0, hi = tal->size;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (tal->edges[mid].t > t_cur) hi = mid;
        else                            lo = mid + 1;
    }
    return lo;
}

/* Write `nwalkers` rows of `walker_len` ints each (space separated, one
 * row per line) to `path`. Dead-end walkers have WALKER_DEAD_END_PAD
 * sentinels in their trailing path slots. */
int log_write(const char* path, const int* paths, int nwalkers, int walker_len);

#endif /* GRAPH_IO_H */
