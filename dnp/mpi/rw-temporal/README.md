# rw-temporal

Distributed random walks on partitioned graphs, implemented with MPI + igraph.

Each MPI rank owns a vertex-disjoint partition of the input graph plus a
routing table that names the rank-owner of every cross-partition neighbour.
Walkers step inside their host partition until they pick a neighbour that
lives elsewhere, at which point they are shipped to the owner rank with
`MPI_Send` and continue from there. After every walker has taken `nsteps`
hops, the gathered paths are written to a single log file.

This repository is the cleaned-up baseline that the temporal-graph work
(see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)) will build on. The
static algorithm in this version is intentionally a faithful reproduction
of the legacy implementation -- only obvious performance bugs have been
fixed; no batching, alias sampling, or temporal logic yet.

## Requirements

- An MPI implementation that provides `mpicc` (OpenMPI / MPICH both work)
- [igraph](https://igraph.org/c/) (tested against 1.0.x; the only 1.0-specific
  call is `igraph_neighbors` with `IGRAPH_LOOPS_TWICE` / `IGRAPH_MULTIPLE`)
- A C99 compiler

## Quick start

```bash
# Build
make

# Single-node smoke test (1 rank, full graph mode)
mpirun -np 1 ./rw facebook 100 80 1

# Distributed run (4 ranks, partitioned mode)
mpirun -np 4 ./rw facebook 1050 80 0

# Multi-host run (uses hostfile)
mpirun -np 8 --hostfile hostfile ./rw facebook 525 80 0
```

Each run writes one log file to `log/<unix_ts>_<dataset>_w<W>_s<S>_p<P>_e<M>.txt`.

## Command-line arguments

```
./rw [dataset] [nwalkers_per_rank] [nsteps] [mode]
```

| Position | Name              | Default    | Meaning                                             |
| -------- | ----------------- | ---------- | --------------------------------------------------- |
| 1        | dataset           | `facebook` | Short name (see below) or a full basename in `data/` |
| 2        | nwalkers_per_rank | `1`        | Number of walkers each rank seeds                   |
| 3        | nsteps            | `80`       | Nodes per walker path (1 random start + `nsteps`-1 hops) |
| 4        | mode              | `0`        | `0` = partitioned, `1` = full graph on every rank   |

Built-in dataset short names:

| short name    | resolved basename                                |
| ------------- | ------------------------------------------------ |
| `facebook`    | `facebook_combined_undirected_connected`         |
| `git`         | `musae_git_edges_undirected.connected`           |
| `twitch`      | `large_twitch_edges_undirected.connected`        |
| `livejournal` | `soc-LiveJournal1_directed.undirected.connected` |

Any other value is treated as a literal basename, so you can drop additional
edgelists into `data/` without touching the source.

## Input data layout

Paths are interpreted relative to the working directory at launch time
(typically the project root).

### Full-graph mode (`mode=1` or `np=1`)

```
data/<basename>.txt
```

A whitespace-separated edgelist: `<src_global> <dst_global>`, one edge per
line. Global node ids may be sparse; they are densified to local ids on load.

### Partitioned mode (`mode=0` and `np>1`)

For `np = P`:

```
data/<P>/<basename>.sub<rank>.txt      # this rank's subgraph edgelist
data/<P>/<basename>.rt<rank>.txt       # this rank's routing table
```

The routing table format is one source-node per line:

```
<src_global> "[(dst0_global, owner_rank0), (dst1_global, owner_rank1), ...]"
```

`src_global` is a node owned by this rank that has at least one neighbour on
another rank; the bracketed list enumerates those out-of-partition neighbours.

## Output log format

Each completed walker is one line of `WALKER_HEADER_INTS + nsteps` integers,
space-separated:

```
<id> <start_ts> <end_ts> <hops_out> <node_0_global> <node_1_global> ... <node_{nsteps-1}_global>
```

| Field        | Meaning                                                       |
| ------------ | ------------------------------------------------------------- |
| `id`         | Walker id (globally unique, `rank * nwalkers_per_rank + i`)   |
| `start_ts`   | `time(NULL)` when the walker was spawned                      |
| `end_ts`     | `time(NULL)` when the walker reached `nsteps`                 |
| `hops_out`   | Number of cross-partition migrations during this walker's life|
| `node_i`     | Global id of the i-th visited node                            |

## Module layout

```
rw-temporal/
├── Makefile
├── README.md
├── docs/
│   └── ARCHITECTURE.md     -- design rationale, data flow, walker lifecycle
├── hostfile                -- list of nodes for distributed runs
├── config.h                -- walker wire layout, defaults, path constants
├── intmap.{c,h}            -- open-addressing int->int hash table
├── routing.{c,h}           -- cross-partition routing table
├── graph_io.{c,h}          -- partition loading + result log writing
├── walker.{c,h}            -- walker state machine + completed-path buffer
├── main.c                  -- MPI driver / main loop
├── data/                   -- input edgelists and partitioned subgraphs
└── log/                    -- one file per run, see "Output log format"
```

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for the responsibility of
each module and the walker lifecycle in detail.

## Roadmap

This baseline is the starting point for a temporal-graph extension. Planned
work, in order:

1. Continuous-time edges `(src, dst, t)` and per-node time-sorted adjacency
2. Temporal walker (`t_cur` field, time-respecting next-hop sampling)
3. Time-window walker bucketing for cache locality (the planned paper's
   main system contribution)
4. Communication batching via `MPI_Alltoallv`
5. Downstream validation on temporal link-prediction benchmarks

See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) "Planned extensions" for
where each piece will hook in.

## License / contact

Original author: smallcat (see source headers). This refactor is a
work-in-progress for a research project; license to be decided.
