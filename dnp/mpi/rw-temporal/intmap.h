/*
 * intmap.h -- minimal open-addressing hash table mapping int -> int.
 *
 * Used as global_id -> local_id (graph_io) and src_global -> route_entry_idx
 * (routing). Keys are non-negative node ids; INT_MIN is reserved as the
 * "empty slot" sentinel.
 */
#ifndef INTMAP_H
#define INTMAP_H

#include <stddef.h>

#define INTMAP_MISS (-1) /* returned by intmap_get when the key is absent */

typedef struct {
    int*   keys;   /* capacity slots; INT_MIN means empty */
    int*   vals;
    size_t cap;    /* always a power of two */
    size_t size;
} intmap_t;

void intmap_init(intmap_t* m, size_t initial_cap);
void intmap_free(intmap_t* m);

/* Returns the stored value, or INTMAP_MISS if `key` is not present. */
int  intmap_get(const intmap_t* m, int key);

/* Insert or overwrite. */
void intmap_put(intmap_t* m, int key, int val);

#endif /* INTMAP_H */
