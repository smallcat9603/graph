/*
 * node_scheduler.h -- per-current-node walker grouping.
 *
 * Walkers currently positioned at local node v are kept together in
 * buckets[v]. Each round, the driver pops *any* non-empty bucket and
 * advances every walker in it by one step. Within such a batch every
 * walker reads TAL[v] -- the same contiguous adjacency array -- so the
 * cache line(s) loaded by the first walker are hot for the rest of the
 * batch. This is the static-graph batching idea (KnightKing,
 * ThunderRW) extended to the time-respecting setting.
 *
 * Hot (power-law hub) nodes naturally accumulate many walkers per round
 * which amortises the TAL load; tail nodes carry one walker each.
 *
 * Active-bucket tracking:
 *   `active[]` is a dense list of node indices whose bucket is non-empty.
 *   `on_active[v]` is the position of v in `active[]`, or -1 if absent.
 *   Insertion / pop are both O(1).
 */
#ifndef NODE_SCHEDULER_H
#define NODE_SCHEDULER_H

#include "walker.h"

typedef struct {
    walker_t** walkers;
    int        count;
    int        cap;
} node_bucket_t;

typedef struct {
    node_bucket_t* buckets;     /* size = num_nodes */
    int            num_nodes;
    int*           active;      /* dense list of non-empty bucket indices */
    int            active_count;
    int*           on_active;   /* on_active[v] = idx in active[], or -1 */
    int            total_alive;
} node_scheduler_t;

void node_scheduler_init(node_scheduler_t* s, int num_nodes);
void node_scheduler_free(node_scheduler_t* s);

/* Insert a walker. Bucket chosen from w->cur_local. */
void node_scheduler_insert(node_scheduler_t* s, walker_t* w);

/* Extract every walker in some non-empty bucket. Returns the walker
 * pointer array (malloc'd; caller frees) and its length via *out_count.
 * Returns NULL with *out_count == 0 if the scheduler is empty. */
walker_t** node_scheduler_pop_any(node_scheduler_t* s, int* out_count);

static inline int node_scheduler_empty(const node_scheduler_t* s) {
    return s->total_alive == 0;
}

#endif /* NODE_SCHEDULER_H */
