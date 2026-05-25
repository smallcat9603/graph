#include "graph_io.h"

#include <stdio.h>
#include <stdlib.h>

void partition_init(partition_t* p) {
    p->l2g    = NULL;
    p->nnodes = 0;
    intmap_init(&p->g2l, 1024);
}

void partition_free(partition_t* p) {
    igraph_destroy(&p->graph);
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

int partition_load_edgelist(partition_t* p, const char* path, bool directed) {
    FILE* fp = fopen(path, "r");
    if (!fp) {
        fprintf(stderr, "partition_load_edgelist: cannot open %s\n", path);
        return -1;
    }

    igraph_vector_int_t edges;
    igraph_vector_int_init(&edges, 0);

    size_t l2g_cap = 0;
    int src, dst;
    while (fscanf(fp, "%d %d", &src, &dst) == 2) {
        int sl = intern_global(p, &l2g_cap, src);
        int dl = intern_global(p, &l2g_cap, dst);
        igraph_vector_int_push_back(&edges, sl);
        igraph_vector_int_push_back(&edges, dl);
    }
    fclose(fp);

    /* Tighten the l2g array to exact size. */
    p->l2g = (int*) realloc(p->l2g, sizeof(int) * p->nnodes);

    igraph_create(&p->graph, &edges, p->nnodes,
                  directed ? IGRAPH_DIRECTED : IGRAPH_UNDIRECTED);
    igraph_vector_int_destroy(&edges);
    return 0;
}

void partition_assert_connected(const partition_t* p) {
    igraph_bool_t connected;
    igraph_is_connected(&p->graph, &connected, IGRAPH_WEAK);
    if (connected) {
        printf("Graph connected: |V|=%lld |E|=%lld\n",
               (long long) igraph_vcount(&p->graph),
               (long long) igraph_ecount(&p->graph));
    } else {
        fprintf(stderr, "Graph is not weakly connected; aborting.\n");
        exit(EXIT_FAILURE);
    }
}

int log_write(const char* path, const int* paths, int nwalkers, int walker_len) {
    FILE* fp = fopen(path, "w");
    if (!fp) {
        fprintf(stderr, "log_write: cannot open %s\n", path);
        return -1;
    }
    for (int w = 0; w < nwalkers; w++) {
        const int* row = paths + (size_t) w * walker_len;
        for (int j = 0; j < walker_len; j++) {
            fprintf(fp, "%d ", row[j]);
        }
        fputc('\n', fp);
    }
    fclose(fp);
    return 0;
}
