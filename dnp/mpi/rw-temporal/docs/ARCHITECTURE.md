# Architecture

This document describes the internal design of `rw-temporal`: how the modules
fit together, how a walker moves through the system, and which design
decisions are deliberate (and why). For build instructions and user-facing
documentation, see [../README.md](../README.md).

## Context

A simple random walk samples a path on a graph. Doing this in a distributed
fashion has two interesting twists:

1. **The graph is too large to live on one machine, so we partition it.**
   Each MPI rank owns a vertex-disjoint subgraph.
2. **A walker is a state, not a piece of data.** When a walker picks a
   neighbour that lives on another rank, the walker itself must travel to
   that rank as a message; the receiving rank then continues the walk as if
   the walker had been local all along.

The original implementation (see commit history before the refactor) was a
prototype. This baseline cleans up the prototype into a small,
single-responsibility set of modules and fixes several obvious correctness
and performance bugs (see "Lessons from the original prototype" below). The
behaviour is unchanged: same algorithm, same wire format, same output log
format.

## High-level data flow

```
                      ┌─────────────────────────────────┐
                      │           main.c                │
                      │  (per-rank driver + MPI loop)   │
                      └──────────────┬──────────────────┘
                                     │ owns
              ┌──────────────────────┼──────────────────────┐
              │                      │                      │
              ▼                      ▼                      ▼
        partition_t              routing_t              path_buf_t
        (graph_io.{c,h})         (routing.{c,h})        (walker.{c,h})
              │                      │                      ▲
              │ uses                 │ uses                 │ push on
              ▼                      ▼                      │ completion
                       intmap_t  (intmap.{c,h})             │
              │                      │                      │
              └──────────┬───────────┘                      │
                         │                                  │
                         ▼                                  │
                    walker_t                                │
                    (walker.{c,h})  ─────────────────────────┘
                       ▲
                       │ MPI_Send / MPI_Recv
                       │  (walker buffer travels between ranks)
                       ▼
                    other ranks
```

## Module responsibilities

| Module        | Owns                                    | Knows about            |
| ------------- | --------------------------------------- | ---------------------- |
| `config.h`    | Constants, walker wire layout macros    | nothing                |
| `intmap`      | int -> int hash table                   | nothing                |
| `routing`     | Cross-partition routing table           | `intmap`               |
| `graph_io`    | Partition loading, ID mapping, log I/O  | `intmap`, igraph       |
| `walker`      | Walker state machine, path collector    | `partition`, `routing` |
| `main.c`      | Argument parsing, MPI loop, file naming | everything             |

The dependency graph is intentionally a DAG -- no header includes any of
the modules sitting above it in the table.

## Key data structures

### Walker (wire format)

A walker is exchanged as a flat `int[]`. The layout is fixed across the
network, so endianness is the user's problem (all hosts in `hostfile` are
assumed homogeneous).

```
offset  field           meaning
------  ---------       ------------------------------------------
   0    id              globally unique walker id
   1    start_ts        time(NULL) at spawn
   2    end_ts          time(NULL) at completion (0 until then)
   3    hops_out        cross-partition migrations so far
   4..  path nodes      global node ids visited, in order
```

`config.h` provides `WALKER_HEADER_INTS = 4` and accessor macros
(`WALKER_ID(buf)`, `WALKER_HOPS_OUT(buf)`, etc.). The total buffer
capacity is `WALKER_HEADER_INTS + nsteps`.

In memory, the running walker carries the wire buffer plus a cached
`cur_local` (the local id of the node the walker is currently sitting
on). The cache is **not** sent across the wire -- the receiver re-derives
it from the last path entry via `g2l`.

### partition_t (per-rank subgraph)

```c
struct partition_t {
    igraph_t graph;     // the loaded igraph on dense local ids 0..nnodes-1
    int*     l2g;       // l2g[local] = global
    int      nnodes;
    intmap_t g2l;       // g2l[global] = local
}
```

The bidirectional ID mapping is the price of using igraph (which insists on
dense vertex ids 0..n-1) over an input edgelist that uses sparse global ids.
`l2g` is an array (constant-time lookup, indexed by local id); `g2l` is a
hash table (the new global -> local direction needed only at walker arrival
time).

### routing_t (cross-partition lookup)

```c
struct route_entry_t {
    int  src_global;
    int* peers;     // [d0_global, p0_rank, d1_global, p1_rank, ...]
    int  npairs;
}
struct routing_t {
    route_entry_t* entries;
    intmap_t       index;   // src_global -> idx into entries
}
```

The flat-pair representation for `peers` mirrors the on-disk file format
and is small enough that hot routing entries fit in cache. Lookup is O(1)
through `index`.

## Walker lifecycle

Each rank runs an identical loop. The state machine for a single walker is:

```
   spawn (this rank)            adopt (received from peer)
        │                                │
        ▼                                ▼
   walker_step ◄─── CONTINUE ────── walker_step
        │                                │
        ├── DONE ──► walker_finalize ─► path_buf_push
        │
        └── MIGRATE ──► MPI_Send ──► (peer rank's MPI_Recv loop)
```

`walker_step` returns one of three codes per call:

- `WALKER_STEP_CONTINUE` -- still inside this partition; caller loops.
- `WALKER_STEP_MIGRATE` -- the chosen next hop lives elsewhere; caller
  `MPI_Send`s the buffer and drops the walker locally.
- `WALKER_STEP_DONE` -- the path is full; caller finalises and stores.

The first call (when `len == WALKER_HEADER_INTS`) picks a uniformly random
starting node from this partition. Subsequent calls hop one node, choosing
uniformly from `local_neighbours ∪ remote_neighbours`. Critically, the
walker **caches its current local id between calls**, so no global-to-local
reverse lookup is needed on the hot path (only on arrival from `MPI_Recv`).

## Main loop on each rank

```c
seed_rng(rank);
load partition + routing
spawn nwalkers_per_rank initial walkers   // each runs to DONE or MIGRATE
while (global_done < total_walkers) {
    MPI_Iprobe TAG_WALKER
    if a walker arrived: walker_adopt + drive
    MPI_Allreduce paths.nwalkers -> global_done
}
gather + write log
```

A few subtleties:

- **Termination detection** is done by `MPI_Allreduce`-ing the count of
  completed walkers every iteration. This is a poll loop -- correct but
  wasteful. Replacing it with Dijkstra-Scholten or a `MPI_Iallreduce`
  every K spins is on the optimisation list.
- **`MPI_Send` is intentionally blocking.** The original code's comment
  explains: non-blocking sends create a race where a single `MPI_Recv` can
  absorb multiple in-flight `MPI_Isend`s, breaking the per-walker handoff.
  The blocking send is correct but obviously the dominant per-walker cost;
  fixing this is the point of the future `MPI_Alltoallv` batching path.
- **The receive side handles "arrived already full" walkers.** If the
  sender's last appended hop happened to fill the path, the walker is
  shipped at full length and the receiver finalises it without taking a
  step. This is rare but legal.

## Lessons from the original prototype

The refactor preserves algorithmic behaviour while fixing seven concrete
issues in the original code:

1. **Per-step global-to-local reverse lookup**
   Original: a linear scan of `node_map[]` on every step (`O(nnodes)`).
   New: walker caches `cur_local`; only the one-time lookup on arrival uses
   the `g2l` hash.

2. **Per-step route table linear search**
   Original: `get_rt()` scanned `dict[]` (`O(rt_size)`).
   New: `routing_lookup` is `O(1)` via `intmap`.

3. **Quadratic node-id densification**
   Original: `map_nodes_in_edgelist()` was `O(nnodes^2)` because each new
   id triggered a full scan of `node_map[]`, and the file was read twice.
   New: a hash-map-backed single pass into an `igraph_vector_int_t`, then
   `igraph_create` builds the graph in memory. No `.x.txt` temp file.

4. **Per-step `realloc` of walker buffer**
   Original: walker buffer grew by one int per step.
   New: full capacity (`WALKER_HEADER_INTS + nsteps`) is allocated once,
   at spawn or adopt. Receivers also grow the inbound buffer to full size
   so subsequent steps need no realloc.

5. **Per-walker `realloc` of completed-paths buffer**
   Original: `paths` grew by one walker at a time.
   New: `path_buf_t` doubles capacity, so total reallocs are O(log n).

6. **Same RNG seed across ranks**
   Original: `srand(time(NULL))`. Ranks initialised in the same second got
   identical streams.
   New: `srand(t ^ rank * 2654435761u)` mixes a Knuth multiplicative hash
   of the rank into the seed.

7. **Hard abort on malformed routing-table line**
   Original: `parseLine` called `exit(0)` on empty or malformed brackets.
   New: warn and skip the line.

## Configuration and tunables

All compile-time knobs live in `config.h`:

| Macro                | Purpose                                       |
| -------------------- | --------------------------------------------- |
| `WALKER_HEADER_INTS` | Number of header ints in the wire format      |
| `DEFAULT_DATASET`    | Default for argv[1]                           |
| `DEFAULT_NWALKERS`   | Default for argv[2]                           |
| `DEFAULT_NSTEPS`     | Default for argv[3]                           |
| `DEFAULT_MODE`       | Default for argv[4] (0 = partitioned)         |
| `DATA_DIR`           | Root of input data (default `data`)           |
| `LOG_DIR`            | Root of output logs (default `log`)           |
| `TAG_WALKER`         | MPI message tag for in-flight walkers         |

Runtime parameters are positional; see [../README.md](../README.md).

## Planned extensions

The temporal-graph work will land in two waves; this section is a forward
reference for where each piece hooks in.

### Wave 1 -- temporal data model and walker

| Change                              | Where it lands                       |
| ----------------------------------- | ------------------------------------ |
| Edge format `(src, dst, t)`         | `graph_io.c` load path               |
| Per-node time-sorted adjacency (TAL)| New `tal.{c,h}` or inside `partition_t` |
| `t_cur` field in walker             | `config.h` (extend header); `walker.c` |
| Time-respecting `pick_next_hop`     | `walker.c` (binary-search on TAL)    |
| Walker termination on dead-end      | `walker_step` return-code change     |
| Time-bucketed routing table         | `routing.c` schema update            |
| Per-dataset short-name update       | `main.c` `dataset_basename`          |

### Wave 2 -- system optimisations

| Change                              | Where it lands                       |
| ----------------------------------- | ------------------------------------ |
| Time-window walker bucketing        | New `frontier.{c,h}`; replaces the   |
|                                     | per-walker `drive_walker` loop       |
| Communication batching via Alltoallv | `main.c` main loop                  |
| Optional alias-method sampler       | `walker.c` `pick_next_hop`           |
| Throttled termination detection     | `main.c` main loop                   |

The pre-existing `partition_t` / `routing_t` boundary is preserved through
both waves -- only the internals (TAL, time index) and the walker contract
(adds `t_cur`) change. Downstream callers (the main loop, the log writer)
should compile against the new structs with at most a couple of additional
positional fields.

## See also

- [../README.md](../README.md) -- user-facing build/run instructions
- [`config.h`](../config.h) -- single source of truth for wire format and defaults
