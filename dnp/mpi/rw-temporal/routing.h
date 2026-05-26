/*
 * routing.h -- per-rank routing table for cross-partition walker hops.
 *
 * Each entry says: "for source-node `src_global`, the global graph has
 * `npeers` neighbours that live OUTSIDE this rank's partition; peer i has
 * global id `peers[i].dst_global`, lives on rank `peers[i].dst_proc`, and
 * the edge between them carries timestamp `peers[i].t`."
 *
 * `peers` is sorted by `t` ascending so a temporal lookup
 * "first remote neighbour with t > t_cur" is an O(log npeers) binary
 * search.
 *
 * On-disk format -- one source per line:
 *
 *     2-tuple legacy:  <src> "[(dst0, proc0), (dst1, proc1), ...]"
 *     3-tuple temporal: <src> "[(dst0, proc0, t0), (dst1, proc1, t1), ...]"
 *
 * 2-tuple rows have their timestamps synthesised via synth_timestamp()
 * so they line up with synthesised local-edge timestamps.
 */
#ifndef ROUTING_H
#define ROUTING_H

#include "intmap.h"

typedef struct {
    int dst_global;
    int dst_proc;
    int t;
} route_peer_t;

typedef struct {
    int           src_global;
    route_peer_t* peers;   /* size = npeers; sorted by t ascending */
    int           npeers;
} route_entry_t;

typedef struct {
    route_entry_t* entries;
    int            nentries;
    intmap_t       index;   /* src_global -> idx into entries[] */
} routing_t;

void routing_init(routing_t* r);
void routing_free(routing_t* r);

/* Load the routing table. Returns 0 on success, -1 on file-open failure.
 * Malformed lines are skipped with a warning. */
int routing_load(routing_t* r, const char* path);

/* O(1) lookup. Returns NULL if `src_global` has no out-of-partition
 * neighbours. */
const route_entry_t* routing_lookup(const routing_t* r, int src_global);

/* Index of the first peer with t > t_cur, or re->npeers if none. */
static inline int routing_upper_bound(const route_entry_t* re, int t_cur) {
    int lo = 0, hi = re->npeers;
    while (lo < hi) {
        int mid = (lo + hi) >> 1;
        if (re->peers[mid].t > t_cur) hi = mid;
        else                            lo = mid + 1;
    }
    return lo;
}

#endif /* ROUTING_H */
