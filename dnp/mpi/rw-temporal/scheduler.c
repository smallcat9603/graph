#include "scheduler.h"

#include <stdlib.h>
#include <string.h>

#include "config.h"

static int bucket_idx(const scheduler_t* s, int t_cur) {
    int t = (t_cur < 0) ? 0 : t_cur;
    int idx = t / s->delta_t;
    if (idx < 0) idx = 0;
    if (idx >= s->num_buckets) idx = s->num_buckets - 1;
    return idx;
}

void scheduler_init(scheduler_t* s, int delta_t, int t_max) {
    if (t_max < 0) t_max = 0;
    if (delta_t == 0) {
        /* Single bucket spanning the whole time range. */
        s->num_buckets = 1;
        s->delta_t     = (t_max > 0) ? (t_max + 1) : 1;
    } else {
        if (delta_t < 0) delta_t = 1;  /* defensive; main.c gates this */
        s->delta_t     = delta_t;
        s->num_buckets = t_max / delta_t + 1;
        if (s->num_buckets < 1) s->num_buckets = 1;
    }
    s->cursor      = 0;
    s->total_alive = 0;
    s->buckets     = (bucket_t*) calloc(s->num_buckets, sizeof(bucket_t));
}

void scheduler_free(scheduler_t* s) {
    if (s->buckets) {
        for (int i = 0; i < s->num_buckets; i++) {
            /* Any walker pointers still in buckets are leaked; the caller
             * should have drained the scheduler before destroying. */
            free(s->buckets[i].walkers);
        }
        free(s->buckets);
    }
    s->buckets     = NULL;
    s->num_buckets = 0;
    s->cursor      = 0;
    s->total_alive = 0;
}

static void bucket_push(bucket_t* b, walker_t* w) {
    if (b->count == b->cap) {
        b->cap = b->cap ? b->cap * 2 : 8;
        b->walkers = (walker_t**) realloc(b->walkers, sizeof(walker_t*) * b->cap);
    }
    b->walkers[b->count++] = w;
}

void scheduler_insert(scheduler_t* s, walker_t* w) {
    int b = bucket_idx(s, WALKER_TCUR(w->buf));
    bucket_push(&s->buckets[b], w);
    if (b < s->cursor) s->cursor = b;
    s->total_alive++;
}

walker_t** scheduler_pop_earliest(scheduler_t* s, int* out_count) {
    while (s->cursor < s->num_buckets && s->buckets[s->cursor].count == 0) {
        s->cursor++;
    }
    if (s->cursor >= s->num_buckets) {
        *out_count = 0;
        return NULL;
    }
    bucket_t* b = &s->buckets[s->cursor];
    *out_count = b->count;
    walker_t** arr = b->walkers;
    s->total_alive -= b->count;
    /* Detach storage; the bucket starts fresh for any new inserts. */
    b->walkers = NULL;
    b->count   = 0;
    b->cap     = 0;
    /* Don't advance cursor: new inserts in this same bucket should still
     * be processed in the next round. The next call's while-loop will
     * advance past it if it stays empty. */
    return arr;
}
