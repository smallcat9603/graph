#include "embed.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static inline float sigmoidf(float x) {
    if (x >  30.0f) return 1.0f;
    if (x < -30.0f) return 0.0f;
    return 1.0f / (1.0f + expf(-x));
}

/* Deterministic per-rank LCG so negative sampling does not perturb the global
 * rand() stream used by the walker. */
static inline unsigned lcg(unsigned* s) {
    *s = (*s) * 1103515245u + 12345u;
    return (*s >> 16) & 0x7fff;
}
static inline int rand_local(embed_t* e) {
    /* combine two 15-bit draws to cover n up to ~10^9 */
    unsigned r = (lcg(&e->rng) << 15) ^ lcg(&e->rng);
    return (int) (r % (unsigned) e->n);
}

void embed_init(embed_t* e, int n, int d, double lr, unsigned seed) {
    e->n = n; e->d = d; e->lr = lr; e->rng = seed ? seed : 1u;
    e->in      = (float*) malloc(sizeof(float) * (size_t) n * d);
    e->out     = (float*) calloc((size_t) n * d, sizeof(float));
    e->scratch = (float*) malloc(sizeof(float) * d);
    /* Small uniform init in [-0.5/d, 0.5/d), like word2vec. */
    for (size_t i = 0; i < (size_t) n * d; i++) {
        e->in[i] = ((float) (lcg(&e->rng)) / 32768.0f - 0.5f) / (float) d;
    }
}

void embed_free(embed_t* e) {
    free(e->in); free(e->out); free(e->scratch);
    e->in = e->out = e->scratch = NULL;
    e->n = e->d = 0;
}

static inline float dot(const float* a, const float* b, int d) {
    float s = 0.0f;
    for (int i = 0; i < d; i++) s += a[i] * b[i];
    return s;
}

void embed_train_pair(embed_t* e, int center, int context, int K, double w_neg) {
    int d = e->d;
    float lr = (float) e->lr;
    float* zc = e->in  + (size_t) center  * d;
    float* vx = e->out + (size_t) context * d;
    float* gc = e->scratch;                 /* accumulated center gradient */

    /* positive */
    float g = sigmoidf(dot(zc, vx, d)) - 1.0f;
    for (int i = 0; i < d; i++) gc[i] = g * vx[i];
    for (int i = 0; i < d; i++) vx[i] -= lr * g * zc[i];

    /* K shard-local negatives */
    for (int k = 0; k < K; k++) {
        int neg = rand_local(e);
        if (neg == context) continue;
        float* vn = e->out + (size_t) neg * d;
        float gn = (float) (sigmoidf(dot(zc, vn, d)) * w_neg);
        for (int i = 0; i < d; i++) gc[i] += gn * vn[i];
        for (int i = 0; i < d; i++) vn[i] -= lr * gn * zc[i];
    }

    /* apply accumulated center gradient last */
    for (int i = 0; i < d; i++) zc[i] -= lr * gc[i];
}

void embed_dump(const embed_t* e, const int* l2g, const char* path) {
    FILE* fp = fopen(path, "w");
    if (!fp) { perror("embed_dump"); return; }
    for (int v = 0; v < e->n; v++) {
        fprintf(fp, "%d", l2g[v]);
        const float* z = e->in + (size_t) v * e->d;
        for (int i = 0; i < e->d; i++) fprintf(fp, " %.6g", z[i]);
        fputc('\n', fp);
    }
    fclose(fp);
}
