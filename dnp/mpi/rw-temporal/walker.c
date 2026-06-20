#include "walker.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

void walker_spawn(walker_t* w, int id, int max_steps, const partition_t* part) {
    w->cap_ints  = WALKER_HEADER_INTS + max_steps;
    w->buf       = (int*) malloc(sizeof(int) * w->cap_ints);
    w->len       = WALKER_HEADER_INTS;
    w->cur_local = -1;
    WALKER_ID(w->buf)       = id;
    WALKER_START_TS(w->buf) = (int) time(NULL);
    WALKER_END_TS(w->buf)   = 0;
    WALKER_HOPS_OUT(w->buf) = 0;
    WALKER_TCUR(w->buf)     = WALKER_INITIAL_TCUR;

    /* Pick a random starting node eagerly so cur_local is valid before
     * the walker reaches any scheduler. */
    if (part->nnodes > 0) {
        int local = rand() % part->nnodes;
        w->cur_local = local;
        w->buf[w->len++] = part->l2g[local];
    }
    /* start the local-run ring with the spawn node */
    w->emb_rlen = 0;
    if (w->cur_local >= 0) { w->emb_ring[0] = w->cur_local; w->emb_rlen = 1; }
}

void walker_adopt(walker_t* w, int* recv_buf, int recv_len, int max_steps,
                  const partition_t* part) {
    w->cap_ints = WALKER_HEADER_INTS + max_steps;
    w->buf      = (int*) malloc(sizeof(int) * w->cap_ints);
    memcpy(w->buf, recv_buf, sizeof(int) * recv_len);
    free(recv_buf);
    w->len = recv_len;

    int last_global = w->buf[w->len - 1];
    int local = intmap_get(&part->g2l, last_global);
    if (local == INTMAP_MISS) {
        fprintf(stderr,
                "walker_adopt: global id %d not found in this partition\n",
                last_global);
        exit(EXIT_FAILURE);
    }
    w->cur_local = local;
    /* fresh local run on this rank (cross-boundary pairs are dropped) */
    w->emb_rlen = 0;
    if (w->cur_local >= 0) { w->emb_ring[0] = w->cur_local; w->emb_rlen = 1; }
}

/* Pick a uniformly random neighbour with t > t_cur, drawing jointly from
 * local edges (TAL[cur_local]) and remote edges (routing entry for
 * cur_global). Returns 0 on success, -1 if no valid edge exists.
 *
 *   local hop:  *out_next_local / *out_next_global / *out_next_t set,
 *               *out_dst_rank = -1
 *   remote hop: *out_next_global / *out_dst_rank / *out_next_t set,
 *               *out_next_local = -1
 */
/* STATIC_WALK env (lazy-read): ignore the t>t_cur constraint and sample from
 * ALL neighbours -- a DeepWalk-style static-walk baseline on the same engine /
 * partition, to isolate the value of the temporal (time-respecting) constraint
 * (DistGER is unbuildable here -- Intel MKL/x86 -- so this is the static ref). */
static int g_static_walk = -1;

static int pick_next_hop(const partition_t* part, const routing_t* routing,
                         int cur_local, int cur_global, int t_cur,
                         int* out_next_local, int* out_next_global,
                         int* out_dst_rank, int* out_next_t) {
    if (g_static_walk < 0) {
        const char* s = getenv("STATIC_WALK");
        g_static_walk = (s && atoi(s) > 0) ? 1 : 0;
    }
    const tal_t* tal = &part->tals[cur_local];
    int local_lo = g_static_walk ? 0 : tal_upper_bound(tal, t_cur);
    int n_local = tal->size - local_lo;

    const route_entry_t* re = routing_lookup(routing, cur_global);
    int remote_lo = 0, n_remote = 0;
    if (re) {
        remote_lo = g_static_walk ? 0 : routing_upper_bound(re, t_cur);
        n_remote  = re->npeers - remote_lo;
    }

    int total = n_local + n_remote;
    if (total == 0) return -1;  /* dead end */

    int pick = rand() % total;

    *out_next_local  = -1;
    *out_next_global = -1;
    *out_dst_rank    = -1;
    *out_next_t      = -1;

    if (pick < n_local) {
        const tal_edge_t* e = &tal->edges[local_lo + pick];
        *out_next_local  = e->neighbor_local;
        *out_next_global = part->l2g[e->neighbor_local];
        *out_next_t      = e->t;
    } else {
        const route_peer_t* p = &re->peers[remote_lo + (pick - n_local)];
        *out_next_global = p->dst_global;
        *out_dst_rank    = p->dst_proc;
        *out_next_t      = p->t;
    }
    return 0;
}

/* ---- E1 instrumentation (de-risking experiment, research_plan_v3.md §8).
 * Count every adjacent (center, context) pair a walk produces and how many
 * of them cross a partition boundary. A boundary-crossing pair is exactly a
 * remote hop, so this measures the fraction of skip-gram pairs that would
 * need cross-rank embedding traffic under co-sharding. Process-local; the
 * caller reduces across ranks. */
static long e1_pairs_total = 0;
static long e1_pairs_cross = 0;

void walker_e1_reset(void) { e1_pairs_total = 0; e1_pairs_cross = 0; }
void walker_e1_get(long* total, long* cross) {
    *total = e1_pairs_total;
    *cross = e1_pairs_cross;
}

int walker_step(walker_t* w, const partition_t* part, const routing_t* routing,
                int* out_dst_rank) {
    if (w->len >= w->cap_ints) return WALKER_STEP_DONE;
    /* Empty partition (nnodes==0, e.g. METIS produced an empty part at high
     * rank counts): the walker never got a start node (cur_local stayed -1).
     * It cannot walk -- terminate instead of dereferencing tals[-1]. */
    if (w->cur_local < 0) return WALKER_STEP_DEAD_END;

    /* The starting node was already placed by walker_spawn / walker_adopt,
     * so every walker_step call takes a real edge. */
    int cur_local  = w->cur_local;
    int cur_global = w->buf[w->len - 1];
    int t_cur      = WALKER_TCUR(w->buf);

    int next_local, next_global, dst_rank, next_t;
    if (pick_next_hop(part, routing, cur_local, cur_global, t_cur,
                      &next_local, &next_global, &dst_rank, &next_t) != 0) {
        return WALKER_STEP_DEAD_END;
    }

    w->buf[w->len++]    = next_global;
    WALKER_TCUR(w->buf) = next_t;

    /* E1: this hop produced one adjacent pair; flag it if it crossed ranks
     * (counted here, before the path-full DONE return, so terminal crossing
     * hops are not missed). */
    e1_pairs_total++;
    if (dst_rank != -1) e1_pairs_cross++;

    /* Path full: walker finalised here even if last hop crossed partitions
     * (the boundary node is recorded but the walker is not migrated). */
    if (w->len >= w->cap_ints) return WALKER_STEP_DONE;

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
    /* Pad unused path slots for dead-end walkers so the on-disk row width
     * stays fixed at cap_ints ints. */
    for (int i = w->len; i < w->cap_ints; i++) {
        w->buf[i] = WALKER_DEAD_END_PAD;
    }
}

void walker_destroy(walker_t* w) {
    free(w->buf);
    w->buf       = NULL;
    w->len       = 0;
    w->cap_ints  = 0;
    w->cur_local = -1;
}

walker_t* walker_create_spawn(int id, int max_steps, const partition_t* part) {
    walker_t* w = (walker_t*) malloc(sizeof(walker_t));
    walker_spawn(w, id, max_steps, part);
    return w;
}

walker_t* walker_create_adopt(int* recv_buf, int recv_len, int max_steps,
                              const partition_t* part) {
    walker_t* w = (walker_t*) malloc(sizeof(walker_t));
    walker_adopt(w, recv_buf, recv_len, max_steps, part);
    return w;
}

void walker_free(walker_t* w) {
    walker_destroy(w);
    free(w);
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
