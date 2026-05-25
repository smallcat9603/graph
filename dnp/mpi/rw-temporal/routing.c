#include "routing.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* Parse one line of the form:
 *     <src> "[(d0, p0), (d1, p1), ...]"
 *
 * Returns 0 on success and fills *src_out / *peers_out / *npairs_out.
 * *peers_out is malloc'd; caller frees.
 * Returns -1 if the line is malformed; nothing is allocated in that case.
 */
static int parse_line(const char* line, int* src_out, int** peers_out, int* npairs_out) {
    if (sscanf(line, "%d", src_out) != 1) return -1;

    const char* lb = strchr(line, '[');
    const char* rb = strchr(line, ']');
    if (!lb || !rb || rb <= lb + 1) return -1;

    /* First pass: count '(' between brackets to size the output array. */
    int npairs = 0;
    for (const char* p = lb; p < rb; p++) {
        if (*p == '(') npairs++;
    }
    if (npairs == 0) return -1;

    int* peers = (int*) malloc(sizeof(int) * 2 * npairs);
    int written = 0;
    const char* cursor = lb;

    while (written < 2 * npairs) {
        const char* lp = strchr(cursor, '(');
        if (!lp || lp >= rb) { free(peers); return -1; }
        int dst, proc;
        if (sscanf(lp, "(%d, %d)", &dst, &proc) != 2) { free(peers); return -1; }
        peers[written++] = dst;
        peers[written++] = proc;
        cursor = strchr(lp, ')');
        if (!cursor) { free(peers); return -1; }
        cursor++;
    }

    *peers_out  = peers;
    *npairs_out = npairs;
    return 0;
}

int routing_load(routing_t* r, const char* path) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "routing_load: cannot open %s\n", path);
        return -1;
    }

    size_t cap = 256;
    r->entries  = (route_entry_t*) malloc(sizeof(route_entry_t) * cap);
    r->nentries = 0;

    char*  line   = NULL;
    size_t bufsz  = 0;

    while (getline(&line, &bufsz, fp) != -1) {
        int  src, npairs;
        int* peers = NULL;
        if (parse_line(line, &src, &peers, &npairs) != 0) {
            fprintf(stderr, "routing_load: skipping malformed line in %s\n", path);
            continue;
        }
        if ((size_t) r->nentries == cap) {
            cap *= 2;
            r->entries = (route_entry_t*) realloc(r->entries, sizeof(route_entry_t) * cap);
        }
        r->entries[r->nentries].src_global = src;
        r->entries[r->nentries].peers      = peers;
        r->entries[r->nentries].npairs     = npairs;
        intmap_put(&r->index, src, r->nentries);
        r->nentries++;
    }

    free(line);
    fclose(fp);
    return 0;
}

const route_entry_t* routing_lookup(const routing_t* r, int src_global) {
    int idx = intmap_get(&r->index, src_global);
    if (idx == INTMAP_MISS) return NULL;
    return &r->entries[idx];
}
