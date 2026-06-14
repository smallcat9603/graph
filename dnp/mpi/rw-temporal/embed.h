/*
 * embed.h -- co-located node-embedding table + skip-gram (SGNS) training.
 *
 * M2 increment 1 (research_plan_v3.md §10): each rank owns the embeddings of
 * its local nodes (co-shard), indexed by LOCAL node id. Training happens in the
 * walk loop on pairs within a local run; negatives are sampled SHARD-LOCAL.
 * Cross-partition pairs are not trained in this increment (added later via the
 * migration-piggyback / periodic exchange).
 *
 * Enabled at runtime by the EMBED_DIM env var (see main.c); off by default so
 * the sampler's existing behaviour is unchanged.
 */
#ifndef EMBED_H
#define EMBED_H

#include <stddef.h>

typedef struct {
    int     n;        /* number of local nodes (embedding rows)        */
    int     d;        /* embedding dimension                           */
    float*  in;       /* n*d  input ("center") vectors -- the output   */
    float*  out;      /* n*d  output ("context") vectors               */
    float*  scratch;  /* d    gradient accumulator for the center      */
    double  lr;       /* learning rate                                 */
    unsigned rng;     /* per-rank RNG state for negative sampling      */
} embed_t;

/* Allocate tables for n local nodes of dimension d; small random init. */
void embed_init(embed_t* e, int n, int d, double lr, unsigned seed);
void embed_free(embed_t* e);

/* One SGNS update for the positive pair (center, context) -- both LOCAL node
 * indices -- with K negatives sampled uniformly from the local shard. w_neg
 * scales the negative gradient (the importance-weight correction; pass 1.0 to
 * disable). */
void embed_train_pair(embed_t* e, int center, int context, int K, double w_neg);

/* Write this rank's shard as "<global_id> v0 v1 ... v(d-1)" lines, using the
 * partition's local-to-global map l2g (length n). */
void embed_dump(const embed_t* e, const int* l2g, const char* path);

#endif /* EMBED_H */
