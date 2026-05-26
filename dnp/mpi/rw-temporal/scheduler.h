/*
 * scheduler.h -- time-window walker bucketing scheduler.
 *
 * Each walker is placed in bucket `t_cur / delta_t`. The driver always
 * pops the earliest non-empty bucket and advances all its walkers by one
 * step; new t_cur values are re-bucketed for the next round.
 *
 * Modes (selected by delta_t value passed to scheduler_init):
 *
 *     delta_t == 0   single bucket   -> "naive batching" (all walkers in
 *                                       one round-robin pool, no time
 *                                       grouping; static-style batching)
 *     delta_t  > 0   N buckets       -> "time-window batching" (the paper
 *                                       contribution; smaller delta_t =
 *                                       finer time grouping)
 *
 * Drive-to-death (one-walker-at-a-time) is handled outside this module
 * by selecting a different main-loop branch in main.c.
 */
#ifndef SCHEDULER_H
#define SCHEDULER_H

#include "walker.h"

typedef struct {
    walker_t** walkers;
    int        count;
    int        cap;
} bucket_t;

typedef struct {
    bucket_t* buckets;
    int       num_buckets;
    int       delta_t;       /* effective bucket width in time units */
    int       cursor;        /* earliest possibly-non-empty bucket index */
    int       total_alive;   /* sum of bucket counts */
} scheduler_t;

/* Initialise the scheduler.
 *   delta_t == 0 -> single bucket (naive batching)
 *   delta_t  > 0 -> ceil((t_max+1)/delta_t) buckets
 */
void scheduler_init(scheduler_t* s, int delta_t, int t_max);
void scheduler_free(scheduler_t* s);

/* Insert a walker; bucket chosen from WALKER_TCUR(w->buf). Negative
 * t_cur (walker spawned but no edge taken yet) maps to bucket 0. */
void scheduler_insert(scheduler_t* s, walker_t* w);

/* Atomically extract every walker in the earliest non-empty bucket.
 * Returns a malloc'd array of walker_t* (length *out_count); caller frees
 * the array. Returns NULL with *out_count == 0 if the scheduler is empty.
 *
 * The bucket is left empty after this call so subsequent inserts at the
 * same bucket index land in fresh storage. */
walker_t** scheduler_pop_earliest(scheduler_t* s, int* out_count);

static inline int scheduler_empty(const scheduler_t* s) {
    return s->total_alive == 0;
}

#endif /* SCHEDULER_H */
