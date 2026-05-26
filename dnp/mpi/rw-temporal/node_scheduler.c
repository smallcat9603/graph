#include "node_scheduler.h"

#include <stdlib.h>
#include <string.h>

void node_scheduler_init(node_scheduler_t* s, int num_nodes) {
    s->num_nodes    = num_nodes;
    s->buckets      = (node_bucket_t*) calloc(num_nodes, sizeof(node_bucket_t));
    s->active       = (int*) malloc(sizeof(int) * num_nodes);
    s->active_count = 0;
    s->on_active    = (int*) malloc(sizeof(int) * num_nodes);
    for (int i = 0; i < num_nodes; i++) s->on_active[i] = -1;
    s->total_alive  = 0;
}

void node_scheduler_free(node_scheduler_t* s) {
    if (s->buckets) {
        for (int i = 0; i < s->num_nodes; i++) free(s->buckets[i].walkers);
        free(s->buckets);
    }
    free(s->active);
    free(s->on_active);
    s->buckets      = NULL;
    s->active       = NULL;
    s->on_active    = NULL;
    s->num_nodes    = 0;
    s->active_count = 0;
    s->total_alive  = 0;
}

static void bucket_push(node_bucket_t* b, walker_t* w) {
    if (b->count == b->cap) {
        b->cap = b->cap ? b->cap * 2 : 8;
        b->walkers = (walker_t**) realloc(b->walkers, sizeof(walker_t*) * b->cap);
    }
    b->walkers[b->count++] = w;
}

void node_scheduler_insert(node_scheduler_t* s, walker_t* w) {
    int v = w->cur_local;
    /* cur_local should always be set by walker_spawn / walker_adopt before
     * insertion; defensively clamp to 0. */
    if (v < 0 || v >= s->num_nodes) v = 0;

    node_bucket_t* b = &s->buckets[v];
    if (b->count == 0 && s->on_active[v] < 0) {
        s->on_active[v] = s->active_count;
        s->active[s->active_count++] = v;
    }
    bucket_push(b, w);
    s->total_alive++;
}

walker_t** node_scheduler_pop_any(node_scheduler_t* s, int* out_count) {
    if (s->active_count == 0) {
        *out_count = 0;
        return NULL;
    }
    /* Pop the most-recently added active node (LIFO). New inserts during
     * the batch may have warmed nearby cache lines, and LIFO has no extra
     * book-keeping. */
    int idx = --s->active_count;
    int v   = s->active[idx];
    s->on_active[v] = -1;

    node_bucket_t* b = &s->buckets[v];
    *out_count = b->count;
    walker_t** arr = b->walkers;
    s->total_alive -= b->count;

    /* Detach storage; the bucket may be repopulated later. */
    b->walkers = NULL;
    b->count   = 0;
    b->cap     = 0;
    return arr;
}
