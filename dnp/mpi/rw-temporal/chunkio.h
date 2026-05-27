/*
 * chunkio.h -- resolve a logical file path to its on-disk chunk(s).
 *
 * Large text files are split at line boundaries into
 *   <logical>.part000, <logical>.part001, ...
 * each smaller than MAX_CHUNK_BYTES (see config.h).
 *
 * Resolution rule:
 *   - if `logical` exists as a regular file  -> a single chunk [logical]
 *   - else                                   -> contiguous .partNNN chunks
 *                                               (stop at first missing index)
 *
 * This lets readers treat split and unsplit files identically.
 */
#ifndef CHUNKIO_H
#define CHUNKIO_H

/* Returns a malloc'd array of malloc'd path strings; sets *count.
 * Caller frees with free_chunks. *count == 0 means nothing was found. */
char** resolve_chunks(const char* logical, int* count);

void free_chunks(char** chunks, int count);

#endif /* CHUNKIO_H */
