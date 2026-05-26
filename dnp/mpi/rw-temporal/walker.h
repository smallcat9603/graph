/*
 * walker.h -- temporal walker state machine + completed-path collector.
 *
 * A walker is a fixed-size int buffer (see config.h for the wire layout)
 * plus a cached local-id of the node it currently sits on. The buffer is
 * allocated once at spawn / adopt time; no per-step reallocs.
 *
 * Temporal semantics: each walker carries `t_cur` in its header. A step
 * samples uniformly from the union of local edges (TAL) and remote edges
 * (routing table) whose timestamps are strictly greater than `t_cur`. If
 * no such edge exists, the walker terminates early (DEAD_END).
 */
#ifndef WALKER_H
#define WALKER_H

#include "config.h"
#include "graph_io.h"
#include "routing.h"

typedef struct {
    int* buf;        /* malloc'd, size = cap_ints */
    int  len;        /* valid ints in buf (header + path so far) */
    int  cap_ints;   /* WALKER_HEADER_INTS + max_steps */
    int  cur_local;  /* local id of current node (-1 before first step) */
} walker_t;

/* Step return codes. */
#define WALKER_STEP_CONTINUE 0  /* stayed local, keep stepping            */
#define WALKER_STEP_MIGRATE  1  /* must be MPI_Send'd to *out_dst_rank    */
#define WALKER_STEP_DONE     2  /* path reached max_steps                 */
#define WALKER_STEP_DEAD_END 3  /* no valid future edge -- terminate      */

/* Allocate a fresh walker buffer of capacity (WALKER_HEADER_INTS + max_steps),
 * stamp id / start_ts, zero hops_out / end_ts, and set t_cur to
 * WALKER_INITIAL_TCUR so the first hop accepts any edge. */
void walker_spawn(walker_t* w, int id, int max_steps);

/* Take ownership of an MPI-received buffer of `recv_len` ints, growing it
 * to full capacity so subsequent steps don't realloc. Resolves cur_local
 * via the partition's g2l hash. `recv_buf` is freed by this function. */
void walker_adopt(walker_t* w, int* recv_buf, int recv_len, int max_steps,
                  const partition_t* part);

/* Advance the walker by exactly one node.
 *
 * - First call (len == WALKER_HEADER_INTS) picks a random starting node
 *   and does not take an edge; t_cur stays at its spawn value.
 * - Later calls find one neighbour (local or remote) with t > t_cur and
 *   take that edge; t_cur is updated to the edge's timestamp.
 *
 * Returns one of WALKER_STEP_*. If MIGRATE is returned, *out_dst_rank is
 * filled with the target rank and the caller must MPI_Send the buffer. */
int  walker_step(walker_t* w, const partition_t* part, const routing_t* routing,
                 int* out_dst_rank);

/* Stamp end_ts and pad any unused path slots with WALKER_DEAD_END_PAD. */
void walker_finalize(walker_t* w);

/* Free the walker buffer. */
void walker_destroy(walker_t* w);


/*
 * path_buf -- growable collection of completed walker buffers. Each slot
 * is `walker_len = WALKER_HEADER_INTS + max_steps` ints. Capacity doubles
 * on overflow, so total reallocs are O(log n).
 */
typedef struct {
    int* data;
    int  nwalkers;
    int  cap;
    int  walker_len;
} path_buf_t;

void path_buf_init(path_buf_t* pb, int walker_len);
void path_buf_free(path_buf_t* pb);
void path_buf_push(path_buf_t* pb, const int* walker_buf);

#endif /* WALKER_H */
