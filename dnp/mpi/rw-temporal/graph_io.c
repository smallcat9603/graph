#include "graph_io.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "chunkio.h"
#include "config.h"

void partition_init(partition_t* p) {
    p->tals   = NULL;
    p->l2g    = NULL;
    p->nnodes = 0;
    p->t_min  = 0;
    p->t_max  = 0;
    intmap_init(&p->g2l, 1024);
}

void partition_free(partition_t* p) {
    if (p->tals) {
        for (int i = 0; i < p->nnodes; i++) free(p->tals[i].edges);
        free(p->tals);
        p->tals = NULL;
    }
    free(p->l2g);
    p->l2g    = NULL;
    p->nnodes = 0;
    intmap_free(&p->g2l);
}

/* Look up `global` in g2l; if absent, assign the next free local id and
 * extend the l2g array (with exponential growth). */
static int intern_global(partition_t* p, size_t* l2g_cap, int global) {
    int local = intmap_get(&p->g2l, global);
    if (local != INTMAP_MISS) return local;

    if ((size_t) p->nnodes == *l2g_cap) {
        *l2g_cap = (*l2g_cap == 0) ? 1024 : (*l2g_cap * 2);
        p->l2g = (int*) realloc(p->l2g, sizeof(int) * (*l2g_cap));
    }
    local = p->nnodes++;
    p->l2g[local] = global;
    intmap_put(&p->g2l, global, local);
    return local;
}

/* Helper: append (neighbor, t) to a growable tal_edge_t buffer. */
typedef struct { tal_edge_t* edges; int size; int cap; } edge_vec_t;

static void edge_vec_push(edge_vec_t* v, int neighbor, int t) {
    if (v->size == v->cap) {
        v->cap = v->cap ? v->cap * 2 : 4;
        v->edges = (tal_edge_t*) realloc(v->edges, sizeof(tal_edge_t) * v->cap);
    }
    v->edges[v->size].neighbor_local = neighbor;
    v->edges[v->size].t = t;
    v->size++;
}

static int cmp_tal_edge_by_t(const void* a, const void* b) {
    int ta = ((const tal_edge_t*) a)->t;
    int tb = ((const tal_edge_t*) b)->t;
    return (ta > tb) - (ta < tb);
}

int partition_load_edgelist(partition_t* p, const char* path) {
    int nchunks;
    char** chunks = resolve_chunks(path, &nchunks);
    if (nchunks == 0) {
        fprintf(stderr, "partition_load_edgelist: cannot find %s (or %s.part000)\n",
                path, path);
        free_chunks(chunks, nchunks);
        return -1;
    }

    /* Edge list of (src_local, dst_local, t); we densify ids while reading
     * and build per-node TALs after the full pass. */
    edge_vec_t* per_node = NULL;
    int per_node_cap = 0;
    size_t l2g_cap = 0;

    char line[512];
    int two_col = 0, three_col = 0;
    int t_min = 0, t_max = 0, t_seen = 0;

    for (int c = 0; c < nchunks; c++) {
        FILE* fp = fopen(chunks[c], "r");
        if (!fp) {
            fprintf(stderr, "partition_load_edgelist: cannot open chunk %s\n", chunks[c]);
            free_chunks(chunks, nchunks);
            free(per_node);
            return -1;
        }
        while (fgets(line, sizeof(line), fp)) {
            int src_g, dst_g, t;
            int n = sscanf(line, "%d %d %d", &src_g, &dst_g, &t);
            if (n == 2) {
                t = synth_timestamp(src_g, dst_g);
                two_col++;
            } else if (n == 3) {
                three_col++;
            } else {
                continue;  /* skip blank / malformed */
            }
            if (!t_seen) { t_min = t_max = t; t_seen = 1; }
            else { if (t < t_min) t_min = t; if (t > t_max) t_max = t; }

            int src_l = intern_global(p, &l2g_cap, src_g);
            int dst_l = intern_global(p, &l2g_cap, dst_g);

            /* Grow per_node vector array if needed */
            while (per_node_cap <= src_l || per_node_cap <= dst_l) {
                int new_cap = per_node_cap ? per_node_cap * 2 : 1024;
                per_node = (edge_vec_t*) realloc(per_node, sizeof(edge_vec_t) * new_cap);
                for (int i = per_node_cap; i < new_cap; i++) {
                    per_node[i].edges = NULL;
                    per_node[i].size  = 0;
                    per_node[i].cap   = 0;
                }
                per_node_cap = new_cap;
            }

            /* Undirected: both endpoints know each other. */
            edge_vec_push(&per_node[src_l], dst_l, t);
            edge_vec_push(&per_node[dst_l], src_l, t);
        }
        fclose(fp);
    }
    free_chunks(chunks, nchunks);

    if (two_col && three_col) {
        fprintf(stderr,
                "partition_load_edgelist: %s mixes 2-col and 3-col rows "
                "(%d and %d resp.); synthesising for 2-col, reading for 3-col\n",
                path, two_col, three_col);
    }

    p->t_min = t_min;
    p->t_max = t_max;

    /* Tighten l2g */
    p->l2g = (int*) realloc(p->l2g, sizeof(int) * p->nnodes);

    /* Allocate tals and copy per-node edges (sorted by t) */
    p->tals = (tal_t*) malloc(sizeof(tal_t) * p->nnodes);
    for (int i = 0; i < p->nnodes; i++) {
        edge_vec_t* v = &per_node[i];
        qsort(v->edges, v->size, sizeof(tal_edge_t), cmp_tal_edge_by_t);
        /* Move ownership to tal_t (trim to exact size). */
        if (v->cap != v->size) {
            v->edges = (tal_edge_t*) realloc(v->edges, sizeof(tal_edge_t) * v->size);
        }
        p->tals[i].edges = v->edges;
        p->tals[i].size  = v->size;
    }
    free(per_node);
    return 0;
}

int partition_ensure_node(partition_t* p, int global) {
    int local = intmap_get(&p->g2l, global);
    if (local != INTMAP_MISS) return local;
    /* Boundary node with no local edges: extend l2g + tals. */
    local = p->nnodes++;
    p->l2g  = (int*)   realloc(p->l2g,  sizeof(int)   * p->nnodes);
    p->tals = (tal_t*) realloc(p->tals, sizeof(tal_t) * p->nnodes);
    p->l2g[local] = global;
    p->tals[local].edges = NULL;
    p->tals[local].size  = 0;
    intmap_put(&p->g2l, global, local);
    return local;
}

/* Format one walker row into `buf` (with trailing -1 compression) and
 * return the byte length written (excluding the NUL). */
static int format_row(char* buf, int cap, const int* row, int walker_len) {
    /* Collapse the trailing run of dead-end padding (-1) to a single
     * marker. Path node ids are non-negative, so a -1 can only be
     * padding; the t_cur field (index 4) may be -1 but is always
     * followed by the start node, so it is never part of the trailing
     * run. A single trailing -1 still signals "walker dead-ended". */
    int last = walker_len - 1;
    while (last >= 0 && row[last] == WALKER_DEAD_END_PAD) last--;

    int len = 0;
    if (last == walker_len - 1) {
        for (int j = 0; j < walker_len; j++)
            len += snprintf(buf + len, cap - len, "%d ", row[j]);
    } else {
        for (int j = 0; j <= last; j++)
            len += snprintf(buf + len, cap - len, "%d ", row[j]);
        len += snprintf(buf + len, cap - len, "%d ", WALKER_DEAD_END_PAD);
    }
    if (len < cap - 1) buf[len++] = '\n';
    buf[len] = '\0';
    return len;
}

int log_write(const char* path, const int* paths, int nwalkers, int walker_len) {
    /* Write to <path>.part000, .part001, ... rolling over to a new part at
     * a row boundary whenever the current part would exceed MAX_CHUNK_BYTES.
     * If only one part is produced, rename it to <path> (single file). */
    const int rowcap = walker_len * 12 + 16;   /* worst-case row text width */
    char* rowbuf = (char*) malloc(rowcap);

    char part_path[600];
    int  part_idx = 0;
    long part_bytes = 0;

    /* Remove any stale output from a previous run with this exact name
     * (single file + all contiguous parts) so leftover higher parts can't
     * be mistaken for current data by the chunk reader. */
    remove(path);
    for (int i = 0; ; i++) {
        char stale[600];
        snprintf(stale, sizeof(stale), "%s.part%03d", path, i);
        if (remove(stale) != 0) break;
    }

    snprintf(part_path, sizeof(part_path), "%s.part%03d", path, part_idx);
    FILE* fp = fopen(part_path, "w");
    if (!fp) {
        fprintf(stderr, "log_write: cannot open %s\n", part_path);
        free(rowbuf);
        return -1;
    }

    for (int w = 0; w < nwalkers; w++) {
        const int* row = paths + (size_t) w * walker_len;
        int rlen = format_row(rowbuf, rowcap, row, walker_len);

        if (part_bytes + rlen > MAX_CHUNK_BYTES && part_bytes > 0) {
            fclose(fp);
            part_idx++;
            part_bytes = 0;
            snprintf(part_path, sizeof(part_path), "%s.part%03d", path, part_idx);
            fp = fopen(part_path, "w");
            if (!fp) {
                fprintf(stderr, "log_write: cannot open %s\n", part_path);
                free(rowbuf);
                return -1;
            }
        }
        fwrite(rowbuf, 1, rlen, fp);
        part_bytes += rlen;
    }
    fclose(fp);
    free(rowbuf);

    /* Single part -> present as a plain <path>; multi-part -> keep .partNNN. */
    if (part_idx == 0) {
        char p0[600];
        snprintf(p0, sizeof(p0), "%s.part000", path);
        rename(p0, path);
    }
    return 0;
}
