/*
 * routing.h -- per-rank routing table for cross-partition walker hops.
 *
 * Each entry says: "for source-node `src_global`, the global graph has
 * `npairs` neighbours that live OUTSIDE this rank's partition; the i-th
 * such neighbour has global id `peers[2*i]` and lives on rank
 * `peers[2*i+1]`."
 *
 * The on-disk file format (one src per line) is:
 *
 *     <src_global> "[(dst0, proc0), (dst1, proc1), ...]"
 */
#ifndef ROUTING_H
#define ROUTING_H

#include "intmap.h"

typedef struct {
    int  src_global;
    int* peers;    /* flat [dst0, proc0, dst1, proc1, ...]; 2*npairs ints */
    int  npairs;
} route_entry_t;

typedef struct {
    route_entry_t* entries;
    int            nentries;
    intmap_t       index;     /* src_global -> idx into entries[] */
} routing_t;

void routing_init(routing_t* r);
void routing_free(routing_t* r);

/* Returns 0 on success, -1 on file-open failure. Malformed lines are
 * skipped with a warning rather than aborting the program. */
int routing_load(routing_t* r, const char* path);

/* O(1) lookup. Returns NULL if `src_global` has no out-of-partition neighbours. */
const route_entry_t* routing_lookup(const routing_t* r, int src_global);

#endif /* ROUTING_H */
