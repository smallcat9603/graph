#include "intmap.h"

#include <limits.h>
#include <stdint.h>
#include <stdlib.h>

#define EMPTY_KEY INT_MIN
#define LOAD_NUM  7   /* trigger resize when size/cap > LOAD_NUM/LOAD_DEN */
#define LOAD_DEN  10

static size_t next_pow2(size_t n) {
    size_t p = 1;
    while (p < n) p <<= 1;
    return p;
}

static uint32_t hash_int(int key) {
    /* Knuth multiplicative hash; good enough for sequential node ids. */
    return (uint32_t) key * 2654435761u;
}

static void alloc_table(intmap_t* m, size_t cap) {
    m->cap  = cap;
    m->size = 0;
    m->keys = (int*) malloc(sizeof(int) * cap);
    m->vals = (int*) malloc(sizeof(int) * cap);
    for (size_t i = 0; i < cap; i++) m->keys[i] = EMPTY_KEY;
}

static void insert_no_resize(intmap_t* m, int key, int val) {
    size_t mask = m->cap - 1;
    size_t i = hash_int(key) & mask;
    while (m->keys[i] != EMPTY_KEY) {
        if (m->keys[i] == key) { m->vals[i] = val; return; }
        i = (i + 1) & mask;
    }
    m->keys[i] = key;
    m->vals[i] = val;
    m->size++;
}

static void resize(intmap_t* m) {
    intmap_t old = *m;
    alloc_table(m, old.cap * 2);
    for (size_t i = 0; i < old.cap; i++) {
        if (old.keys[i] != EMPTY_KEY) {
            insert_no_resize(m, old.keys[i], old.vals[i]);
        }
    }
    free(old.keys);
    free(old.vals);
}

void intmap_init(intmap_t* m, size_t initial_cap) {
    if (initial_cap < 16) initial_cap = 16;
    alloc_table(m, next_pow2(initial_cap));
}

void intmap_free(intmap_t* m) {
    free(m->keys);
    free(m->vals);
    m->keys = NULL;
    m->vals = NULL;
    m->cap  = 0;
    m->size = 0;
}

int intmap_get(const intmap_t* m, int key) {
    size_t mask = m->cap - 1;
    size_t i = hash_int(key) & mask;
    while (m->keys[i] != EMPTY_KEY) {
        if (m->keys[i] == key) return m->vals[i];
        i = (i + 1) & mask;
    }
    return INTMAP_MISS;
}

void intmap_put(intmap_t* m, int key, int val) {
    if ((m->size + 1) * LOAD_DEN > m->cap * LOAD_NUM) {
        resize(m);
    }
    insert_no_resize(m, key, val);
}
