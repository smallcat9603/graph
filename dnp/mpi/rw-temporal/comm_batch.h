/*
 * comm_batch.h -- batched cross-rank walker migration via MPI_Alltoallv.
 *
 * Instead of one blocking MPI_Send per migrating walker, walkers about
 * to migrate are queued in a per-destination outbound buffer. When the
 * caller decides to flush, all ranks exchange their full outbound
 * buffers in a single MPI_Alltoallv collective, then parse the inbound
 * data into walker_t objects.
 *
 * Wire format inside an outbound buffer (and the receive side):
 *
 *     [len_0, w0_data..., len_1, w1_data..., ...]
 *
 * Each chunk starts with the int-length of the walker buffer, followed
 * by that many ints of walker state.
 */
#ifndef COMM_BATCH_H
#define COMM_BATCH_H

#include "walker.h"

typedef struct {
    int* data;        /* flat buffer */
    int  total_ints;  /* number of valid ints */
    int  cap;         /* allocated ints */
    int  nwalkers;    /* count of walkers queued (for stats) */
} outbound_t;

/* Allocate `size` outbound buffers, one per peer rank. */
outbound_t* outbound_array_alloc(int size);
void        outbound_array_free(outbound_t* arr, int size);

/* Queue a walker for delivery to `dst_rank`. Takes a copy of the walker
 * bytes; caller still owns / can free the source. */
void outbound_push(outbound_t* arr, int dst_rank, const int* walker_buf, int walker_len);

/* Total number of queued walkers across all destinations (for stats). */
int outbound_total_pending(const outbound_t* arr, int size);

#endif /* COMM_BATCH_H */
