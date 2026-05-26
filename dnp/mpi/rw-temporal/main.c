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

#include "config.h"
#include "graph_io.h"
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

static void seed_rng(int rank) {
    uint32_t t = (uint32_t) time(NULL);
    uint32_t r = (uint32_t) rank * 2654435761u;
    srand((unsigned int) (t ^ r));
}

/* ------------------------------------------------------------ drive-to-death
 *
 * Legacy Wave 1 main loop: every walker runs to completion (or migration)
 * before the next one starts. Kept as an ablation baseline.
 */

static int drive_walker(walker_t* w, const partition_t* part, const routing_t* routing,
                        path_buf_t* paths) {
    int dst_rank;
    for (;;) {
        int r = walker_step(w, part, routing, &dst_rank);
        if (r == WALKER_STEP_CONTINUE) continue;
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
                           scheduler_t* sched, path_buf_t* paths) {
    for (int i = 0; i < n; i++) {
        walker_t* w = arr[i];
        int dst_rank;
        int r = walker_step(w, part, routing, &dst_rank);
        switch (r) {
            case WALKER_STEP_CONTINUE:
                scheduler_insert(sched, w);
                break;
            case WALKER_STEP_DONE:
            case WALKER_STEP_DEAD_END:
                walker_finalize(w);
                path_buf_push(paths, w->buf);
                walker_free(w);
                break;
            case WALKER_STEP_MIGRATE:
                MPI_Send(w->buf, w->len, MPI_INT, dst_rank, TAG_WALKER, MPI_COMM_WORLD);
                walker_free(w);
                break;
        }
    }
}

static void run_bucketed(const args_t* args, int rank, int total_walkers,
                         const partition_t* part, const routing_t* routing,
                         path_buf_t* paths) {
    int id_start = rank * args->nwalkers_per_rank;

    scheduler_t sched;
    scheduler_init(&sched, args->delta_t, part->t_max);

    for (int i = 0; i < args->nwalkers_per_rank; i++) {
        walker_t* w = walker_create_spawn(id_start + i, args->nsteps, part);
        scheduler_insert(&sched, w);
    }

    int global_done = 0;
    while (global_done < total_walkers) {
        /* 1) Drain one bucket -- single round of work per outer iteration. */
        int n;
        walker_t** arr = scheduler_pop_earliest(&sched, &n);
        if (arr) {
            process_bucket(arr, n, part, routing, &sched, paths);
            free(arr);
        }

        /* 2) Absorb any walker that just arrived from a peer rank. */
        int        flag = 0;
        MPI_Status status;
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_WALKER, MPI_COMM_WORLD, &flag, &status);
        if (flag) {
            int count;
            MPI_Get_count(&status, MPI_INT, &count);
            int* recv = (int*) malloc(sizeof(int) * count);
            MPI_Recv(recv, count, MPI_INT, status.MPI_SOURCE, TAG_WALKER,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            walker_t* w = walker_create_adopt(recv, count, args->nsteps, part);
            if (w->len >= w->cap_ints) {
                walker_finalize(w);
                path_buf_push(paths, w->buf);
                walker_free(w);
            } else {
                scheduler_insert(&sched, w);
            }
        }

        /* 3) Global termination check. */
        MPI_Allreduce(&paths->nwalkers, &global_done, 1, MPI_INT, MPI_SUM,
                      MPI_COMM_WORLD);
    }

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
                                node_scheduler_t* sched, path_buf_t* paths) {
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
                MPI_Send(w->buf, w->len, MPI_INT, dst_rank, TAG_WALKER, MPI_COMM_WORLD);
                walker_free(w);
                break;
        }
    }
}

static void run_node_grouped(const args_t* args, int rank, int total_walkers,
                             const partition_t* part, const routing_t* routing,
                             path_buf_t* paths) {
    int id_start = rank * args->nwalkers_per_rank;

    node_scheduler_t sched;
    node_scheduler_init(&sched, part->nnodes);

    for (int i = 0; i < args->nwalkers_per_rank; i++) {
        walker_t* w = walker_create_spawn(id_start + i, args->nsteps, part);
        node_scheduler_insert(&sched, w);
    }

    int global_done = 0;
    while (global_done < total_walkers) {
        int n;
        walker_t** arr = node_scheduler_pop_any(&sched, &n);
        if (arr) {
            process_node_bucket(arr, n, part, routing, &sched, paths);
            free(arr);
        }

        int        flag = 0;
        MPI_Status status;
        MPI_Iprobe(MPI_ANY_SOURCE, TAG_WALKER, MPI_COMM_WORLD, &flag, &status);
        if (flag) {
            int count;
            MPI_Get_count(&status, MPI_INT, &count);
            int* recv = (int*) malloc(sizeof(int) * count);
            MPI_Recv(recv, count, MPI_INT, status.MPI_SOURCE, TAG_WALKER,
                     MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            walker_t* w = walker_create_adopt(recv, count, args->nsteps, part);
            if (w->len >= w->cap_ints) {
                walker_finalize(w);
                path_buf_push(paths, w->buf);
                walker_free(w);
            } else {
                node_scheduler_insert(&sched, w);
            }
        }

        MPI_Allreduce(&paths->nwalkers, &global_done, 1, MPI_INT, MPI_SUM,
                      MPI_COMM_WORLD);
    }

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
        printf("rank=%d loaded routing %s\n", rank, rt_path);
    }

    const int walker_len = WALKER_HEADER_INTS + args.nsteps;
    path_buf_t paths;
    path_buf_init(&paths, walker_len);

    double t0 = MPI_Wtime();

    if (args.policy == POLICY_NODE) {
        run_node_grouped(&args, rank, total_walkers, &part, &routing, &paths);
    } else if (args.delta_t < 0) {
        run_drive_to_death(&args, rank, total_walkers, &part, &routing, &paths);
    } else {
        run_bucketed(&args, rank, total_walkers, &part, &routing, &paths);
    }

    double t1 = MPI_Wtime();

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
        free(counts);
        free(displs);
        free(all_paths);
    }

    path_buf_free(&paths);
    routing_free(&routing);
    partition_free(&part);
    MPI_Finalize();
    return 0;
}
