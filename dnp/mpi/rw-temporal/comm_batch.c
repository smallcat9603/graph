#include "comm_batch.h"

#include <stdlib.h>
#include <string.h>

outbound_t* outbound_array_alloc(int size) {
    outbound_t* arr = (outbound_t*) calloc(size, sizeof(outbound_t));
    return arr;
}

void outbound_array_free(outbound_t* arr, int size) {
    if (!arr) return;
    for (int i = 0; i < size; i++) free(arr[i].data);
    free(arr);
}

void outbound_push(outbound_t* arr, int dst_rank,
                   const int* walker_buf, int walker_len) {
    outbound_t* b = &arr[dst_rank];
    int needed = 1 + walker_len;
    if (b->total_ints + needed > b->cap) {
        int new_cap = b->cap ? b->cap * 2 : 256;
        while (new_cap < b->total_ints + needed) new_cap *= 2;
        b->data = (int*) realloc(b->data, sizeof(int) * new_cap);
        b->cap  = new_cap;
    }
    b->data[b->total_ints] = walker_len;
    memcpy(b->data + b->total_ints + 1, walker_buf, sizeof(int) * walker_len);
    b->total_ints += needed;
    b->nwalkers++;
}

int outbound_total_pending(const outbound_t* arr, int size) {
    int n = 0;
    for (int i = 0; i < size; i++) n += arr[i].nwalkers;
    return n;
}
