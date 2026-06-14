/*
 * main.c -- MPI driver for distributed continuous-time random walks (CTDW)
 *           on a partitioned graph, with three scheduling modes.
 *
 * Usage:
 *     mpirun -np <P> ./rw [dataset] [nwalkers_per_rank] [nsteps] [mode] [delta_t]
 *
 * Arguments (all optional, see config.h for defaults):
 *     dataset           short name: facebook | git | twitch | livejournal |
 *                       wikipedia | reddit | mooc, or a full basename in data/
 *     nwalkers_per_rank number of walkers each rank seeds
 *     nsteps            number of nodes in each walker's path
 *     mode              0 = partitioned, 1 = full graph per rank
 *     delta_t           scheduling policy:
 *                          < 0  drive-to-death (legacy Wave 1 baseline)
 *                          = 0  single-bucket scheduler (naive batching)
 *                          > 0  N-bucket scheduler with width delta_t
 *                               (time-window batching, the paper's contribution)
 *
 * Outputs one log at log/<unix_ts>_<dataset>_w<W>_s<S>_p<P>_e<M>_dt<D>.txt
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>

#include "comm_batch.h"
#include "config.h"
#include "embed.h"
#include "graph_io.h"
#include "intmap.h"
#include "node_scheduler.h"
#include "routing.h"
#include "scheduler.h"
#include "walker.h"

typedef enum { MODE_PARTITION = 0, MODE_FULL = 1 } run_mode_t;
typedef enum { POLICY_AUTO = 0, POLICY_NODE = 1 } policy_t;

typedef struct {
    char       dataset[64];
    int        nwalkers_per_rank;
    int        nsteps;
    run_mode_t mode;
    int        delta_t;
    policy_t   policy;
} args_t;

static void parse_args(int argc, char** argv, args_t* a) {
    snprintf(a->dataset, sizeof(a->dataset), "%s", DEFAULT_DATASET);
    a->nwalkers_per_rank = DEFAULT_NWALKERS;
    a->nsteps            = DEFAULT_NSTEPS;
    a->mode              = (run_mode_t) DEFAULT_MODE;
    a->delta_t           = DEFAULT_DELTA_T;
    a->policy            = POLICY_AUTO;
    if (argc > 1) snprintf(a->dataset, sizeof(a->dataset), "%s", argv[1]);
    if (argc > 2) a->nwalkers_per_rank = atoi(argv[2]);
    if (argc > 3) a->nsteps            = atoi(argv[3]);
    if (argc > 4) a->mode              = (run_mode_t) atoi(argv[4]);
    if (argc > 5) a->delta_t           = atoi(argv[5]);
    if (argc > 6) a->policy            = (policy_t) atoi(argv[6]);
}

static const char* dataset_basename(const char* short_name) {
    if (!strcmp(short_name, "facebook"))    return "facebook_combined_undirected_connected";
    if (!strcmp(short_name, "git"))         return "musae_git_edges_undirected.connected";
    if (!strcmp(short_name, "twitch"))      return "large_twitch_edges_undirected.connected";
    if (!strcmp(short_name, "livejournal")) return "soc-LiveJournal1_directed.undirected.connected";
    return short_name;
}

static void build_paths(const char* dataset, int rank, int size, run_mode_t mode,
                        char* edge_path, char* rt_path, size_t cap) {
    const char* base = dataset_basename(dataset);
    int partitioned  = (mode == MODE_PARTITION) && (size > 1);
    if (partitioned) {
        snprintf(edge_path, cap, "%s/%d/%s.sub%d.txt", DATA_DIR, size, base, rank);
        snprintf(rt_path,   cap, "%s/%d/%s.rt%d.txt",  DATA_DIR, size, base, rank);
    } else {
        snprintf(edge_path, cap, "%s/%s.txt", DATA_DIR, base);
        rt_path[0] = '\0';
    }
}

/* Base seed: RNG_SEED env if set (reproducible multi-seed runs), else time. */
static uint32_t rng_base(void) {
    const char* s = getenv("RNG_SEED");
    return s ? (uint32_t) atoi(s) : (uint32_t) time(NULL);
}
static void seed_rng(int rank) {
    uint32_t r = (uint32_t) rank * 2654435761u;
    srand((unsigned int) (rng_base() ^ r));
}

/* ----------------------------------------------------------- batched comm
 *
 * Parse a contiguous chunk received from one peer (or any peer's slice of
 * the Alltoallv recvbuf) and adopt every walker into the scheduler.
 * `node_sched` xor `time_sched` may be non-NULL depending on policy. */
static void absorb_recv_chunk(int* chunk, int chunk_len, int max_steps,
                              partition_t* part,
                              node_scheduler_t* node_sched,
                              scheduler_t* time_sched,
                              path_buf_t* paths) {
    int p = 0;
    while (p < chunk_len) {
        int wlen = chunk[p];
        int* wdata = (int*) malloc(sizeof(int) * wlen);
        memcpy(wdata, &chunk[p + 1], sizeof(int) * wlen);
        p += 1 + wlen;

        /* Ensure the incoming node has a local slot (covers boundary
         * nodes whose only edges are cross-partition). */
        int last_global = wdata[wlen - 1];
        partition_ensure_node(part, last_global);

        walker_t* w = walker_create_adopt(wdata, wlen, max_steps, part);
        if (w->len >= w->cap_ints) {
            walker_finalize(w);
            path_buf_push(paths, w->buf);
            walker_free(w);
        } else if (node_sched) {
            node_scheduler_insert(node_sched, w);
        } else if (time_sched) {
            scheduler_insert(time_sched, w);
        } else {
            walker_free(w);
        }
    }
}

/* Collective Alltoallv: every rank flushes all queued outbound walkers,
 * receives walkers destined for itself, and absorbs them. */
static void flush_round(outbound_t* outbound, int size, int max_steps,
                        partition_t* part,
                        node_scheduler_t* node_sched,
                        scheduler_t* time_sched,
                        path_buf_t* paths) {
    int* sendcounts = (int*) malloc(sizeof(int) * size);
    int* sdispls    = (int*) malloc(sizeof(int) * size);
    int total_send = 0;
    for (int j = 0; j < size; j++) {
        sendcounts[j] = outbound[j].total_ints;
        sdispls[j]    = total_send;
        total_send   += sendcounts[j];
    }
    int* sendbuf = (total_send > 0) ? (int*) malloc(sizeof(int) * total_send) : NULL;
    for (int j = 0; j < size; j++) {
        if (sendcounts[j] > 0) {
            memcpy(sendbuf + sdispls[j], outbound[j].data,
                   sizeof(int) * sendcounts[j]);
        }
    }

    int* recvcounts = (int*) malloc(sizeof(int) * size);
    MPI_Alltoall(sendcounts, 1, MPI_INT, recvcounts, 1, MPI_INT, MPI_COMM_WORLD);

    int* rdispls = (int*) malloc(sizeof(int) * size);
    int total_recv = 0;
    for (int j = 0; j < size; j++) {
        rdispls[j]  = total_recv;
        total_recv += recvcounts[j];
    }
    int* recvbuf = (total_recv > 0) ? (int*) malloc(sizeof(int) * total_recv) : NULL;

    MPI_Alltoallv(sendbuf, sendcounts, sdispls, MPI_INT,
                  recvbuf, recvcounts, rdispls, MPI_INT, MPI_COMM_WORLD);

    for (int j = 0; j < size; j++) {
        if (recvcounts[j] > 0) {
            absorb_recv_chunk(recvbuf + rdispls[j], recvcounts[j], max_steps,
                              part, node_sched, time_sched, paths);
        }
    }

    for (int j = 0; j < size; j++) {
        outbound[j].total_ints = 0;
        outbound[j].nwalkers   = 0;
    }

    free(sendcounts); free(sdispls);  free(sendbuf);
    free(recvcounts); free(rdispls);  free(recvbuf);
}

/* ------------------------------------------------------------ drive-to-death
 *
 * Legacy Wave 1 main loop: every walker runs to completion (or migration)
 * before the next one starts. Kept as an ablation baseline.
 */

/* M2 increment 1: co-shard embedding training (off unless EMBED_DIM set). */
static embed_t* g_embed   = NULL;
static int      g_emb_win = 5;   /* skip-gram window */
static int      g_emb_neg = 5;   /* negatives per pair */
static double   g_emb_wneg = 1.0;/* negative importance weight (1 = off) */
static long     g_emb_pairs = 0; /* positive pairs trained so far (for twostage) */
static int      g_emb_mode  = 0; /* 0 = fused (local negs), 1 = two-stage (NOMAD-style) */

/* Two-stage / NOMAD-style embedding-communication for ONE round: a faithful
 * comm pattern (not a separate trainer) that moves the embedding-vector volume
 * a remote-negative design would -- K * pairs_round * (P-1)/P negative vectors
 * fetched and their deltas returned, via MPI_Alltoallv of real d-float rows.
 * Returns the wall-clock spent. Quantifies exactly the communication our fused
 * (shard-local-negative) engine AVOIDS. */
static double twostage_embed_exchange(long pairs_round, int K, int size, int rank,
                                      embed_t* e) {
    /* MUST be called collectively by every rank each round: pairs_round varies
     * per rank, so it cannot gate the collectives (would deadlock). When
     * pairs_round==0 the volume falls to the per=1 floor below. */
    if (!e || size < 2) return 0.0;
    if (pairs_round < 0) pairs_round = 0;
    int d = e->d;
    long remote_neg = (long) ((double) K * (double) pairs_round
                              * (double) (size - 1) / (double) size);
    int per = (int) (remote_neg / (size - 1));         /* ids requested per peer */
    if (per < 1) per = 1;
    if (per > e->n) per = e->n;

    int* scnt = (int*) malloc(sizeof(int) * size);
    int* rcnt = (int*) malloc(sizeof(int) * size);
    int* sdis = (int*) malloc(sizeof(int) * size);
    int* rdis = (int*) malloc(sizeof(int) * size);
    for (int j = 0; j < size; j++) scnt[j] = (j == rank) ? 0 : per;

    double t0 = MPI_Wtime();
    /* `per` is capped by each rank's local node count, so it differs per rank;
     * exchange the real counts before Alltoallv to avoid TRUNCATE. */
    MPI_Alltoall(scnt, 1, MPI_INT, rcnt, 1, MPI_INT, MPI_COMM_WORLD);
    int stot = 0, rtot = 0;
    for (int j = 0; j < size; j++) { sdis[j] = stot; stot += scnt[j]; rdis[j] = rtot; rtot += rcnt[j]; }
    /* 1. request: send `per` random row ids to each peer */
    int* sreq = (int*) malloc(sizeof(int) * (stot ? stot : 1));
    int* rreq = (int*) malloc(sizeof(int) * (rtot ? rtot : 1));
    for (int i = 0; i < stot; i++) sreq[i] = rand() % e->n;
    MPI_Alltoallv(sreq, scnt, sdis, MPI_INT, rreq, rcnt, rdis, MPI_INT, MPI_COMM_WORLD);

    /* 2. reply with the requested d-float rows; 3. return deltas (same volume) */
    int* scntd = (int*) malloc(sizeof(int) * size);
    int* rcntd = (int*) malloc(sizeof(int) * size);
    int* sdisd = (int*) malloc(sizeof(int) * size);
    int* rdisd = (int*) malloc(sizeof(int) * size);
    for (int j = 0; j < size; j++) { scntd[j] = rcnt[j] * d; rcntd[j] = scnt[j] * d; }
    int sdt = 0, rdt = 0;
    for (int j = 0; j < size; j++) { sdisd[j] = sdt; sdt += scntd[j]; rdisd[j] = rdt; rdt += rcntd[j]; }
    float* reply = (float*) malloc(sizeof(float) * (sdt ? sdt : 1));
    float* recvv = (float*) malloc(sizeof(float) * (rdt ? rdt : 1));
    for (int i = 0; i < rtot; i++)
        memcpy(reply + (size_t) i * d, e->in + (size_t) rreq[i] * d, sizeof(float) * d);
    MPI_Alltoallv(reply, scntd, sdisd, MPI_FLOAT, recvv, rcntd, rdisd, MPI_FLOAT, MPI_COMM_WORLD);
    /* delta send-back: same volume in the reverse direction */
    MPI_Alltoallv(recvv, rcntd, rdisd, MPI_FLOAT, reply, scntd, sdisd, MPI_FLOAT, MPI_COMM_WORLD);
    double t = MPI_Wtime() - t0;

    free(scnt); free(rcnt); free(sdis); free(rdis);
    free(sreq); free(rreq);
    free(scntd); free(rcntd); free(sdisd); free(rdisd); free(reply); free(recvv);
    return t;
}

/* Train window pairs within a local run: `cur` (local idx) is the node just
 * stepped to; `ring`/`rlen` hold up to g_emb_win previous LOCAL node indices.
 * Trains (cur, ctx) for each ctx in the ring, then pushes cur. */
static void embed_on_local_step(int* ring, int* rlen, int cur) {
    if (!g_embed) return;
    for (int j = 0; j < *rlen; j++)
        embed_train_pair(g_embed, cur, ring[j], g_emb_neg, g_emb_wneg);
    g_emb_pairs += *rlen;   /* positive pairs trained (drives two-stage comm volume) */
    if (*rlen < g_emb_win) {
        ring[(*rlen)++] = cur;
    } else {
        for (int j = 1; j < g_emb_win; j++) ring[j - 1] = ring[j];
        ring[g_emb_win - 1] = cur;
    }
}

static int drive_walker(walker_t* w, const partition_t* part, const routing_t* routing,
                        path_buf_t* paths) {
    int dst_rank;
    for (;;) {
        int r = walker_step(w, part, routing, &dst_rank);
        if (r == WALKER_STEP_CONTINUE) {
            embed_on_local_step(w->emb_ring, &w->emb_rlen, w->cur_local);
            continue;
        }
        if (r == WALKER_STEP_DONE || r == WALKER_STEP_DEAD_END) {
            walker_finalize(w);
            path_buf_push(paths, w->buf);
            walker_destroy(w);
            return 1;
        }
        MPI_Send(w->buf, w->len, MPI_INT, dst_rank, TAG_WALKER, MPI_COMM_WORLD);
        walker_destroy(w);
        return 0;
    }
}

static void run_drive_to_death(const args_t* args, int rank, int total_walkers,
                               const partition_t* part, const routing_t* routing,
                               path_buf_t* paths) {
    int id_start = rank * args->nwalkers_per_rank;
    int walker_len = WALKER_HEADER_INTS + args->nsteps;
    (void) walker_len;

    for (int i = 0; i < args->nwalkers_per_rank; i++) {
        walker_t w;
        walker_spawn(&w, id_start + i, args->nsteps, part);
        drive_walker(&w, part, routing, paths);
    }

    int global_done = 0;
    while (global_done < total_walkers) {
        int        flag = 0;
        MPI_Status status;
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_WALKER, MPI_COMM_WORLD, &flag, &status);
        if (flag) {
            int count;
            MPI_Get_count(&status, MPI_INT, &count);
            int* recv = (int*) malloc(sizeof(int) * count);
            MPI_Recv(recv, count, MPI_INT, status.MPI_SOURCE, TAG_WALKER,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            walker_t w;
            walker_adopt(&w, recv, count, args->nsteps, part);
            if (w.len >= w.cap_ints) {
                walker_finalize(&w);
                path_buf_push(paths, w.buf);
                walker_destroy(&w);
            } else {
                drive_walker(&w, part, routing, paths);
            }
        }
        MPI_Allreduce(&paths->nwalkers, &global_done, 1, MPI_INT, MPI_SUM,
                      MPI_COMM_WORLD);
    }
}

/* ------------------------------------------------------------ bucket scheduler
 *
 * Wave 2 main loop: walkers are kept in a time-bucketed scheduler. Each
 * iteration drains the earliest non-empty bucket, advances every walker
 * in it by one step, and re-inserts the survivors. Used for both
 * naive batching (delta_t == 0, single bucket) and time-window batching
 * (delta_t > 0).
 */

static void process_bucket(walker_t** arr, int n,
                           const partition_t* part, const routing_t* routing,
                           scheduler_t* sched,
                           outbound_t* outbound,
                           path_buf_t* paths) {
    for (int i = 0; i < n; i++) {
        walker_t* w = arr[i];
        int dst_rank;
        int r = walker_step(w, part, routing, &dst_rank);
        switch (r) {
            case WALKER_STEP_CONTINUE:
                embed_on_local_step(w->emb_ring, &w->emb_rlen, w->cur_local);
                scheduler_insert(sched, w);
                break;
            case WALKER_STEP_DONE:
            case WALKER_STEP_DEAD_END:
                walker_finalize(w);
                path_buf_push(paths, w->buf);
                walker_free(w);
                break;
            case WALKER_STEP_MIGRATE:
                outbound_push(outbound, dst_rank, w->buf, w->len);
                walker_free(w);
                break;
        }
    }
}

static void run_bucketed(const args_t* args, int rank, int size,
                         int total_walkers,
                         partition_t* part, const routing_t* routing,
                         path_buf_t* paths) {
    int id_start = rank * args->nwalkers_per_rank;

    scheduler_t sched;
    scheduler_init(&sched, args->delta_t, part->t_max);

    outbound_t* outbound = outbound_array_alloc(size);

    for (int i = 0; i < args->nwalkers_per_rank; i++) {
        walker_t* w = walker_create_spawn(id_start + i, args->nsteps, part);
        scheduler_insert(&sched, w);
    }

    /* F4 phase timers: separate compute / exchange (Alltoallv) / termination
     * (Allreduce) so the communication fraction can be reported (the headline
     * scaling metric on a real multi-node network). */
    double t_compute = 0, t_exchange = 0, t_allreduce = 0, t_embxchg = 0;
    int global_done = 0;
    while (global_done < total_walkers) {
        double ta = MPI_Wtime();
        long pairs_before = g_emb_pairs;
        /* Drain every local bucket before each Alltoallv flush. */
        while (!scheduler_empty(&sched)) {
            int n;
            walker_t** arr = scheduler_pop_earliest(&sched, &n);
            if (!arr) break;
            process_bucket(arr, n, part, routing, &sched, outbound, paths);
            free(arr);
        }

        double tb = MPI_Wtime();
        flush_round(outbound, size, args->nsteps, part, NULL, &sched, paths);

        double tc = MPI_Wtime();
        /* Two-stage baseline: pay the NOMAD-style remote embedding comm that the
         * fused (local-negative) design avoids -- volume from this round's pairs. */
        if (g_emb_mode == 1 && g_embed)
            t_embxchg += twostage_embed_exchange(g_emb_pairs - pairs_before,
                                                 g_emb_neg, size, rank, g_embed);

        double tcc = MPI_Wtime();
        MPI_Allreduce(&paths->nwalkers, &global_done, 1, MPI_INT, MPI_SUM,
                      MPI_COMM_WORLD);
        double td = MPI_Wtime();
        t_compute += tb - ta; t_exchange += tc - tb;
        t_allreduce += td - tcc;
    }

    /* Report the slowest-rank time per phase (wall-clock determinant). */
    double loc[4] = { t_compute, t_exchange, t_allreduce, t_embxchg }, mx[4];
    MPI_Reduce(loc, mx, 4, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        double tot = mx[0] + mx[1] + mx[2] + mx[3];
        printf("PHASE compute=%.4f exchange=%.4f allreduce=%.4f emb_xchg=%.4f s  "
               "comm_frac=%.1f%% [p=%d mode=%s]\n",
               mx[0], mx[1], mx[2], mx[3],
               tot > 0 ? 100.0 * (mx[1] + mx[2] + mx[3]) / tot : 0.0,
               size, g_emb_mode ? "twostage" : "fused");
    }

    outbound_array_free(outbound, size);
    scheduler_free(&sched);
}

/* ------------------------------------------------------------ node grouping
 *
 * Wave 3 main loop: walkers are bucketed by their CURRENT NODE rather
 * than by t_cur. Each iteration drains one node bucket; all walkers in
 * that batch share the same TAL, giving real cache locality.
 */

static void process_node_bucket(walker_t** arr, int n,
                                const partition_t* part, const routing_t* routing,
                                node_scheduler_t* sched,
                                outbound_t* outbound,
                                path_buf_t* paths) {
    for (int i = 0; i < n; i++) {
        walker_t* w = arr[i];
        int dst_rank;
        int r = walker_step(w, part, routing, &dst_rank);
        switch (r) {
            case WALKER_STEP_CONTINUE:
                node_scheduler_insert(sched, w);
                break;
            case WALKER_STEP_DONE:
            case WALKER_STEP_DEAD_END:
                walker_finalize(w);
                path_buf_push(paths, w->buf);
                walker_free(w);
                break;
            case WALKER_STEP_MIGRATE:
                /* Batched: queue for next Alltoallv flush instead of
                 * issuing a blocking MPI_Send per walker. */
                outbound_push(outbound, dst_rank, w->buf, w->len);
                walker_free(w);
                break;
        }
    }
}

static void run_node_grouped(const args_t* args, int rank, int size,
                             int total_walkers,
                             partition_t* part, const routing_t* routing,
                             path_buf_t* paths) {
    int id_start = rank * args->nwalkers_per_rank;

    node_scheduler_t sched;
    node_scheduler_init(&sched, part->nnodes);

    outbound_t* outbound = outbound_array_alloc(size);

    for (int i = 0; i < args->nwalkers_per_rank; i++) {
        walker_t* w = walker_create_spawn(id_start + i, args->nsteps, part);
        node_scheduler_insert(&sched, w);
    }

    int global_done = 0;
    while (global_done < total_walkers) {
        /* Drain the local scheduler completely before each flush so
         * Alltoallv can carry as many walkers per call as possible. */
        while (!node_scheduler_empty(&sched)) {
            int n;
            walker_t** arr = node_scheduler_pop_any(&sched, &n);
            if (!arr) break;
            process_node_bucket(arr, n, part, routing, &sched, outbound, paths);
            free(arr);
        }

        /* Collective: send all queued walkers, receive everyone else's. */
        flush_round(outbound, size, args->nsteps, part, &sched, NULL, paths);

        /* Inbound walkers (if any) now live in the scheduler; loop again
         * to process them. If every rank's scheduler stays empty AND no
         * one queued anything, the Allreduce below will let us exit. */
        MPI_Allreduce(&paths->nwalkers, &global_done, 1, MPI_INT, MPI_SUM,
                      MPI_COMM_WORLD);
    }

    outbound_array_free(outbound, size);
    node_scheduler_free(&sched);
}

/* ------------------------------------------------------------ main */

int main(int argc, char** argv) {
    args_t args;
    parse_args(argc, argv, &args);

    MPI_Init(NULL, NULL);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    seed_rng(rank);

    const int total_walkers = args.nwalkers_per_rank * size;
    const int id_start      = rank * args.nwalkers_per_rank;

    const char* mode_name;
    if (args.policy == POLICY_NODE) {
        mode_name = "node-grouping";
    } else if (args.delta_t < 0) {
        mode_name = "drive-to-death";
    } else if (args.delta_t == 0) {
        mode_name = "single-bucket";
    } else {
        mode_name = "time-window";
    }
    printf("rank=%d/%d dataset=%s walkers=%d (%d..%d) steps=%d mode=%d sched=%s delta_t=%d policy=%d\n",
           rank, size, args.dataset, args.nwalkers_per_rank,
           id_start, id_start + args.nwalkers_per_rank - 1,
           args.nsteps, (int) args.mode, mode_name, args.delta_t, (int) args.policy);

    char edge_path[512], rt_path[512];
    build_paths(args.dataset, rank, size, args.mode, edge_path, rt_path, sizeof(edge_path));

    partition_t part;
    partition_init(&part);
    if (partition_load_edgelist(&part, edge_path) != 0) {
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    printf("rank=%d loaded %s (|V_local|=%d  t_range=[%d,%d])\n",
           rank, edge_path, part.nnodes, part.t_min, part.t_max);

    routing_t routing;
    routing_init(&routing);
    if (rt_path[0] != '\0') {
        if (routing_load(&routing, rt_path) != 0) {
            fprintf(stderr, "rank=%d failed to load %s\n", rank, rt_path);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        printf("rank=%d loaded routing %s (%d entries)\n",
               rank, rt_path, routing.nentries);

        /* Ensure every routing source has a local slot, even when all of
         * its edges are cross-partition (so the node never appeared in
         * sub<r>.txt). */
        int added = 0;
        for (int i = 0; i < routing.nentries; i++) {
            int src = routing.entries[i].src_global;
            int before = part.nnodes;
            partition_ensure_node(&part, src);
            if (part.nnodes > before) added++;
        }
        if (added > 0) {
            printf("rank=%d added %d boundary-only nodes from routing\n", rank, added);
        }
    }

    const int walker_len = WALKER_HEADER_INTS + args.nsteps;
    path_buf_t paths;
    path_buf_init(&paths, walker_len);

    /* M2 increment 1: enable co-shard embedding training if EMBED_DIM is set.
     * Trains only in the drive-to-death loop (delta_t < 0). */
    embed_t embed_store;
    {
        const char* sdim = getenv("EMBED_DIM");
        if (sdim && atoi(sdim) > 0) {
            int d = atoi(sdim);
            const char* sw  = getenv("EMBED_WIN");  if (sw)  g_emb_win  = atoi(sw);
            if (g_emb_win > 8) g_emb_win = 8;   /* bounded by walker_t.emb_ring */
            const char* sn  = getenv("EMBED_NEG");  if (sn)  g_emb_neg  = atoi(sn);
            const char* swn = getenv("EMBED_WNEG"); if (swn) g_emb_wneg = atof(swn);
            const char* sm  = getenv("EMBED_MODE");
            if (sm && strcmp(sm, "twostage") == 0) g_emb_mode = 1;
            double lr = 0.025;
            const char* slr = getenv("EMBED_LR");   if (slr) lr = atof(slr);
            embed_init(&embed_store, part.nnodes, d, lr,
                       (unsigned) ((rank + 1) * 2654435761u) ^ rng_base());
            g_embed = &embed_store;
            if (rank == 0)
                printf("EMBED: dim=%d win=%d neg=%d wneg=%.3f lr=%.3f "
                       "(co-shard training ON; drive-to-death + batched loops)\n",
                       d, g_emb_win, g_emb_neg, g_emb_wneg, lr);
        }
    }

    walker_e1_reset();   /* E1 de-risking: count cross-rank adjacent pairs */
    double t0 = MPI_Wtime();

    if (args.policy == POLICY_NODE) {
        run_node_grouped(&args, rank, size, total_walkers, &part, &routing, &paths);
    } else if (args.delta_t < 0) {
        run_drive_to_death(&args, rank, total_walkers, &part, &routing, &paths);
    } else {
        run_bucketed(&args, rank, size, total_walkers, &part, &routing, &paths);
    }

    double t1 = MPI_Wtime();

    /* M2 increment 1: dump this rank's co-shard embedding shard for offline
     * temporal-LP evaluation. */
    if (g_embed) {
        char ep[512];
        snprintf(ep, sizeof(ep), "%s/embed_%s_p%d_r%d.txt",
                 LOG_DIR, args.dataset, size, rank);
        embed_dump(g_embed, part.l2g, ep);
        if (rank == 0)
            printf("EMBED: wrote shards %s/embed_%s_p%d_r*.txt\n",
                   LOG_DIR, args.dataset, size);
        embed_free(g_embed);
        g_embed = NULL;
    }

    /* E1 (research_plan_v3.md §8): fraction of adjacent (center, context)
     * pairs that cross a partition boundary == fraction of skip-gram pairs
     * needing cross-rank embedding traffic under co-sharding. Small fraction
     * ==> co-shard premise holds. */
    long e1_local[2], e1_global[2];
    walker_e1_get(&e1_local[0], &e1_local[1]);
    MPI_Reduce(e1_local, e1_global, 2, MPI_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        long tot = e1_global[0], crs = e1_global[1];
        printf("E1 cross-rank pairs: %ld/%ld (%.2f%%) [p=%d]\n",
               crs, tot, tot ? 100.0 * (double) crs / (double) tot : 0.0, size);
    }

    /* Gather and write log */
    int  local_npaths = paths.nwalkers;
    int* counts = NULL;
    int* displs = NULL;
    if (rank == 0) {
        counts = (int*) malloc(sizeof(int) * size);
        displs = (int*) malloc(sizeof(int) * size);
    }
    MPI_Gather(&local_npaths, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    int* all_paths       = NULL;
    int  total_paths_int = 0;
    if (rank == 0) {
        displs[0] = 0;
        counts[0] *= walker_len;
        for (int i = 1; i < size; i++) {
            displs[i] = displs[i - 1] + counts[i - 1];
            counts[i] *= walker_len;
        }
        total_paths_int = displs[size - 1] + counts[size - 1];
        all_paths = (int*) malloc(sizeof(int) * total_paths_int);
    }
    MPI_Gatherv(paths.data, local_npaths * walker_len, MPI_INT,
                all_paths, counts, displs, MPI_INT, 0, MPI_COMM_WORLD);

    /* E1 window-w (gated by env E1_WINDOW=w): exact cross-rank fraction of
     * skip-gram pairs for windows >1. Window-1 (the always-on counter above)
     * equals the migration rate; for w>1 a pair (v_i,v_j) crosses iff its two
     * endpoints are owned by different ranks, which needs a global owner map.
     * Off by default, so normal runs pay nothing. */
    int e1_W = 0;
    { const char* s = getenv("E1_WINDOW"); if (s) e1_W = atoi(s); }
    int* owned_counts = NULL;
    int* owned_displs = NULL;
    int* all_owned    = NULL;
    if (e1_W > 0) {
        if (rank == 0) owned_counts = (int*) malloc(sizeof(int) * size);
        MPI_Gather(&part.nnodes, 1, MPI_INT, owned_counts, 1, MPI_INT,
                   0, MPI_COMM_WORLD);
        int total_owned = 0;
        if (rank == 0) {
            owned_displs = (int*) malloc(sizeof(int) * size);
            owned_displs[0] = 0;
            for (int i = 1; i < size; i++)
                owned_displs[i] = owned_displs[i - 1] + owned_counts[i - 1];
            total_owned = owned_displs[size - 1] + owned_counts[size - 1];
            all_owned = (int*) malloc(sizeof(int) * total_owned);
        }
        MPI_Gatherv(part.l2g, part.nnodes, MPI_INT,
                    all_owned, owned_counts, owned_displs, MPI_INT,
                    0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        char log_path[512];
        snprintf(log_path, sizeof(log_path),
                 "%s/%d_%s_w%d_s%d_p%d_e%d_dt%d_pol%d.txt",
                 LOG_DIR, (int) t1, args.dataset,
                 args.nwalkers_per_rank, args.nsteps, size,
                 (int) args.mode, args.delta_t, (int) args.policy);
        log_write(log_path, all_paths, total_paths_int / walker_len, walker_len);
        printf("wrote %s\n", log_path);
        printf("rank=0 elapsed=%fs sched=%s delta_t=%d total_walkers=%d\n",
               t1 - t0, mode_name, args.delta_t, total_walkers);

        /* E1 window-w: build the owner map and count cross-rank pairs for
         * every skip-gram window up to e1_W, over the gathered full paths. */
        if (e1_W > 0) {
            intmap_t owner;
            intmap_init(&owner, (size_t) (owned_displs[size - 1] +
                                          owned_counts[size - 1]) * 2 + 16);
            for (int r = 0; r < size; r++)
                for (int k = 0; k < owned_counts[r]; k++)
                    intmap_put(&owner, all_owned[owned_displs[r] + k], r);

            long* wt = (long*) calloc(e1_W + 1, sizeof(long)); /* total by gap */
            long* wc = (long*) calloc(e1_W + 1, sizeof(long)); /* cross by gap */
            long miss = 0;
            int  path_cap = walker_len - WALKER_HEADER_INTS;
            int  nwalk    = total_paths_int / walker_len;
            for (int wkr = 0; wkr < nwalk; wkr++) {
                const int* path = all_paths + (size_t) wkr * walker_len
                                  + WALKER_HEADER_INTS;
                int plen = 0;
                while (plen < path_cap && path[plen] != WALKER_DEAD_END_PAD)
                    plen++;
                for (int i = 0; i < plen; i++) {
                    int oi = intmap_get(&owner, path[i]);
                    if (oi == INTMAP_MISS) { miss++; continue; }
                    for (int k = 1; k <= e1_W && i + k < plen; k++) {
                        int oj = intmap_get(&owner, path[i + k]);
                        if (oj == INTMAP_MISS) { miss++; continue; }
                        wt[k]++;
                        if (oi != oj) wc[k]++;
                    }
                }
            }
            long ct = 0, cc = 0;
            for (int w = 1; w <= e1_W; w++) {
                ct += wt[w];
                cc += wc[w];
                printf("E1 window=%d cross-rank pairs: %ld/%ld (%.2f%%) [p=%d]\n",
                       w, cc, ct, ct ? 100.0 * (double) cc / (double) ct : 0.0,
                       size);
            }
            if (miss) printf("E1 window: %ld owner-map misses (unexpected)\n", miss);

            /* M1 / F2-pilot: embedding-communication accounting at the training
             * window e1_W. Empirical inputs: cc = cross-rank positive pairs,
             * ct = total pairs (from the real walks + real partition above).
             * Two-stage / NOMAD-style: every cross positive pair and every
             * remote negative is fetched AND its delta sent back (2 vectors of
             * d float32); negatives are global, so a fraction (P-1)/P land off
             * the center's shard. Fused co-shard: the cross positive pair's
             * gradient rides the migration message already in flight (1 vector),
             * and negatives are sampled shard-local (0 remote, exchange off).
             * Migration of walk state is identical for both, so it is excluded;
             * this is the embedding-comm differentiator only. Bytes, not wall
             * clock — the real fused engine (M2+) is needed for timing. */
            {
                int d = 128, K = 5; double rho = 0.1;
                const char* sd = getenv("E1_DIM"); if (sd) d = atoi(sd);
                const char* sk = getenv("E1_NEG"); if (sk) K = atoi(sk);
                const char* sr = getenv("E1_RHO"); if (sr) rho = atof(sr);
                double vec = (double) d * 4.0;                 /* float32 vector */
                double remote_neg = (double) K * (double) ct
                                    * (double) (size - 1) / (double) size;
                double twostage = (cc + remote_neg) * 2.0 * vec;
                /* fused: piggybacked positive grad (1 vec/cross pair) + the
                 * rho-fraction of negatives still exchanged remotely (2 vec).
                 * rho=0 is the optimistic bound; rho=0.1 is E2's quality point. */
                double fused_lo = (double) cc * vec;                       /* rho=0 */
                double fused    = fused_lo + rho * remote_neg * 2.0 * vec; /* rho   */
                printf("F2 window=%d d=%d K=%d rho=%.2f: two-stage=%.1f MB  "
                       "fused=%.1f MB (%.1fx)  fused@rho0=%.1f MB (%.1fx)  "
                       "[cross=%ld total=%ld remote_neg=%.0f]\n",
                       e1_W, d, K, rho,
                       twostage / 1e6,
                       fused / 1e6,    fused    > 0 ? twostage / fused : 0.0,
                       fused_lo / 1e6, fused_lo > 0 ? twostage / fused_lo : 0.0,
                       cc, ct, remote_neg);
            }

            intmap_free(&owner);
            free(wt);
            free(wc);
        }

        free(counts);
        free(displs);
        free(all_paths);
        free(owned_counts);
        free(owned_displs);
        free(all_owned);
    }

    path_buf_free(&paths);
    routing_free(&routing);
    partition_free(&part);
    MPI_Finalize();
    return 0;
}
