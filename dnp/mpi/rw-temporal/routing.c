#include "routing.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chunkio.h"
#include "config.h"

void routing_init(routing_t* r) {
    r->entries  = NULL;
    r->nentries = 0;
    intmap_init(&r->index, 256);
}

void routing_free(routing_t* r) {
    for (int i = 0; i < r->nentries; i++) {
        free(r->entries[i].peers);
    }
    free(r->entries);
    r->entries  = NULL;
    r->nentries = 0;
    intmap_free(&r->index);
}

/* Portable getline replacement (getline is POSIX-only; under some
 * cross-compilers it is implicitly declared, corrupting the buffer pointer on
 * 64-bit targets). Reads one '\n'-terminated line, growing *buf as needed.
 * Returns line length, or -1 at EOF with nothing read. */
static long read_line(char** buf, size_t* cap, FILE* fp) {
    if (*buf == NULL) { *cap = 256; *buf = (char*) malloc(*cap); }
    size_t len = 0;
    int ch;
    while ((ch = fgetc(fp)) != EOF) {
        if (len + 1 >= *cap) { *cap *= 2; *buf = (char*) realloc(*buf, *cap); }
        if (ch == '\n') break;
        (*buf)[len++] = (char) ch;
    }
    if (ch == EOF && len == 0) return -1;
    (*buf)[len] = '\0';
    return (long) len;
}

static int cmp_peer_by_t(const void* a, const void* b) {
    int ta = ((const route_peer_t*) a)->t;
    int tb = ((const route_peer_t*) b)->t;
    return (ta > tb) - (ta < tb);
}

/* Parse one line of the form:
 *     <src> "[(d0, p0[, t0]), (d1, p1[, t1]), ...]"
 *
 * On success fills *src_out / *peers_out / *npeers_out (peers sorted by t).
 * *peers_out is malloc'd; caller frees. Returns -1 on malformed line. */
static int parse_line(const char* line, int* src_out,
                      route_peer_t** peers_out, int* npeers_out) {
    if (sscanf(line, "%d", src_out) != 1) return -1;
    int src = *src_out;

    const char* lb = strchr(line, '[');
    const char* rb = strchr(line, ']');
    if (!lb || !rb || rb <= lb + 1) return -1;

    int npeers = 0;
    for (const char* p = lb; p < rb; p++) {
        if (*p == '(') npeers++;
    }
    if (npeers == 0) return -1;

    route_peer_t* peers = (route_peer_t*) malloc(sizeof(route_peer_t) * npeers);
    int written = 0;
    const char* cursor = lb;

    while (written < npeers) {
        const char* lp = strchr(cursor, '(');
        if (!lp || lp >= rb) { free(peers); return -1; }

        int dst, proc, t;
        int n = sscanf(lp, "(%d, %d, %d)", &dst, &proc, &t);
        if (n == 3) {
            /* 3-tuple: timestamp provided */
        } else if (sscanf(lp, "(%d, %d)", &dst, &proc) == 2) {
            t = synth_timestamp(src, dst);
        } else {
            free(peers); return -1;
        }

        peers[written].dst_global = dst;
        peers[written].dst_proc   = proc;
        peers[written].t          = t;
        written++;

        cursor = strchr(lp, ')');
        if (!cursor) { free(peers); return -1; }
        cursor++;
    }

    qsort(peers, npeers, sizeof(route_peer_t), cmp_peer_by_t);
    *peers_out  = peers;
    *npeers_out = npeers;
    return 0;
}

int routing_load(routing_t* r, const char* path) {
    int nchunks;
    char** chunks = resolve_chunks(path, &nchunks);
    if (nchunks == 0) {
        fprintf(stderr, "routing_load: cannot find %s (or %s.part000)\n", path, path);
        free_chunks(chunks, nchunks);
        return -1;
    }

    size_t cap = 256;
    r->entries  = (route_entry_t*) malloc(sizeof(route_entry_t) * cap);
    r->nentries = 0;

    char*  line  = NULL;
    size_t bufsz = 0;

    for (int c = 0; c < nchunks; c++) {
        FILE* fp = fopen(chunks[c], "r");
        if (!fp) {
            fprintf(stderr, "routing_load: cannot open chunk %s\n", chunks[c]);
            free(line);
            free_chunks(chunks, nchunks);
            return -1;
        }
        while (read_line(&line, &bufsz, fp) != -1) {
            int           src, npeers;
            route_peer_t* peers = NULL;
            if (parse_line(line, &src, &peers, &npeers) != 0) {
                fprintf(stderr, "routing_load: skipping malformed line in %s\n", chunks[c]);
                continue;
            }
            if ((size_t) r->nentries == cap) {
                cap *= 2;
                r->entries = (route_entry_t*) realloc(r->entries, sizeof(route_entry_t) * cap);
            }
            r->entries[r->nentries].src_global = src;
            r->entries[r->nentries].peers      = peers;
            r->entries[r->nentries].npeers     = npeers;
            intmap_put(&r->index, src, r->nentries);
            r->nentries++;
        }
        fclose(fp);
    }

    free(line);
    free_chunks(chunks, nchunks);
    return 0;
}

const route_entry_t* routing_lookup(const routing_t* r, int src_global) {
    int idx = intmap_get(&r->index, src_global);
    if (idx == INTMAP_MISS) return NULL;
    return &r->entries[idx];
}
