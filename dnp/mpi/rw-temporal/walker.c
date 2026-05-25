#include "walker.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static int walker_capacity(int max_steps) {
    return WALKER_HEADER_INTS + max_steps;
}

void walker_spawn(walker_t* w, int id, int max_steps) {
    int cap = walker_capacity(max_steps);
    w->buf = (int*) malloc(sizeof(int) * cap);
    w->len = WALKER_HEADER_INTS;
    w->cur_local = -1;
    WALKER_ID(w->buf)       = id;
    WALKER_START_TS(w->buf) = (int) time(NULL);
    WALKER_END_TS(w->buf)   = 0;
    WALKER_HOPS_OUT(w->buf) = 0;
}

void walker_adopt(walker_t* w, int* recv_buf, int recv_len, int max_steps,
                  const partition_t* part) {
    int cap = walker_capacity(max_steps);
    w->buf = (int*) malloc(sizeof(int) * cap);
    memcpy(w->buf, recv_buf, sizeof(int) * recv_len);
    free(recv_buf);
    w->len = recv_len;

    /* Last entry in the path is the global node where this walker re-enters. */
    int last_global = w->buf[w->len - 1];
    int local = intmap_get(&part->g2l, last_global);
    if (local == INTMAP_MISS) {
        fprintf(stderr,
                "walker_adopt: global id %d not found in this partition\n",
                last_global);
        exit(EXIT_FAILURE);
    }
    w->cur_local = local;
}

/* Pick a uniformly random neighbour (local or remote) of (cur_local, cur_global).
 * On local hop:  *out_next_local / *out_next_global set, *out_dst_rank = -1
 * On remote hop: *out_next_global / *out_dst_rank set, *out_next_local = -1 */
static void pick_next_hop(const partition_t* part, const routing_t* routing,
                          int cur_local, int cur_global,
                          int* out_next_local, int* out_next_global,
                          int* out_dst_rank) {
    igraph_vector_int_t nbrs;
    igraph_vector_int_init(&nbrs, 0);
    igraph_neighbors(&part->graph, &nbrs, (igraph_integer_t) cur_local, IGRAPH_ALL);
    int n_local = (int) igraph_vector_int_size(&nbrs);

    const route_entry_t* re = routing_lookup(routing, cur_global);
    int n_remote = re ? re->npairs : 0;

    int total = n_local + n_remote;
    int pick  = rand() % total;

    *out_next_local  = -1;
    *out_next_global = -1;
    *out_dst_rank    = -1;

    if (pick < n_local) {
        *out_next_local  = (int) VECTOR(nbrs)[pick];
        *out_next_global = part->l2g[*out_next_local];
    } else {
        int j = (pick - n_local) * 2;
        *out_next_global = re->peers[j];
        *out_dst_rank    = re->peers[j + 1];
    }

    igraph_vector_int_destroy(&nbrs);
}

int walker_step(walker_t* w, const partition_t* part, const routing_t* routing,
                int max_steps, int* out_dst_rank) {
    int cap = walker_capacity(max_steps);
    if (w->len >= cap) return WALKER_STEP_DONE;

    /* First step: pick a random starting node in this partition. */
    if (w->len == WALKER_HEADER_INTS) {
        int local = rand() % part->nnodes;
        w->cur_local = local;
        w->buf[w->len++] = part->l2g[local];
        return (w->len >= cap) ? WALKER_STEP_DONE : WALKER_STEP_CONTINUE;
    }

    /* Subsequent step: hop from the current node. cur_local is cached. */
    int cur_local  = w->cur_local;
    int cur_global = w->buf[w->len - 1];

    int next_local, next_global, dst_rank;
    pick_next_hop(part, routing, cur_local, cur_global,
                  &next_local, &next_global, &dst_rank);

    w->buf[w->len++] = next_global;

    /* Path full: walker finalised here even if last hop crossed partitions
     * (matches the original semantics -- the boundary node is recorded but
     * the walker is not migrated). */
    if (w->len >= cap) return WALKER_STEP_DONE;

    if (dst_rank != -1) {
        WALKER_HOPS_OUT(w->buf)++;
        *out_dst_rank = dst_rank;
        return WALKER_STEP_MIGRATE;
    }
    w->cur_local = next_local;
    return WALKER_STEP_CONTINUE;
}

void walker_finalize(walker_t* w) {
    WALKER_END_TS(w->buf) = (int) time(NULL);
}

void walker_destroy(walker_t* w) {
    free(w->buf);
    w->buf = NULL;
    w->len = 0;
    w->cur_local = -1;
}

/* ---------------------------------------------------------------- path_buf */

void path_buf_init(path_buf_t* pb, int walker_len) {
    pb->walker_len = walker_len;
    pb->nwalkers   = 0;
    pb->cap        = 64;
    pb->data       = (int*) malloc(sizeof(int) * pb->cap * walker_len);
}

void path_buf_free(path_buf_t* pb) {
    free(pb->data);
    pb->data     = NULL;
    pb->nwalkers = 0;
    pb->cap      = 0;
}

void path_buf_push(path_buf_t* pb, const int* walker_buf) {
    if (pb->nwalkers == pb->cap) {
        pb->cap *= 2;
        pb->data = (int*) realloc(pb->data,
                                  sizeof(int) * pb->cap * pb->walker_len);
    }
    memcpy(pb->data + (size_t) pb->nwalkers * pb->walker_len,
           walker_buf,
           sizeof(int) * pb->walker_len);
    pb->nwalkers++;
}
