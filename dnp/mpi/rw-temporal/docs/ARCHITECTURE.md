# Architecture

This document describes the internal design of `rw-temporal`: how the modules
fit together, how a walker moves through the system, and which design
decisions are deliberate (and why). For build instructions and user-facing
documentation, see [../README.md](../README.md); for benchmark numbers see
[../results.md](../results.md).

## Context

`rw-temporal` performs **distributed continuous-time random walks (CTDW)** on
a partitioned temporal graph. Two ideas shape the design:

1. **The graph is partitioned across MPI ranks.** Each rank owns a
   vertex-disjoint subgraph plus a routing table naming the rank-owner of
   every cross-partition neighbour.
2. **A walker is migrating state, not data.** When a walker steps to a node
   owned by another rank, the walker itself travels there as a message and
   the receiving rank continues the walk.

The temporal aspect adds a third constraint:

3. **Time-respecting walks.** Each edge carries a timestamp `t`. A walker
   carries a cursor `t_cur` and may only traverse edges with `t > t_cur`;
   `t_cur` advances to the chosen edge's timestamp. A walker with no valid
   future edge terminates early (a *dead end*).

The project evolved in four waves:

| wave | what landed |
| --- | --- |
| 0.2 (baseline) | static-graph refactor: clean modules, bug fixes, igraph backend |
| Wave 1 | temporal data model (TAL), `t_cur`, time-respecting sampling, igraph dropped |
| Wave 2 | time-bucketed walker scheduler (`scheduler.c`) |
| Wave 3 | node-grouping walker scheduler (`node_scheduler.c`) |
| Wave 4 | batched cross-rank migration via `MPI_Alltoallv` (`comm_batch.c`) + METIS partitioning |

The empirical headline (see results.md): batched communication is **78–347×**
faster than per-walker `MPI_Send`, and once communication is batched the
intra-rank scheduling policy matters <20%.

## High-level data flow

```
                      ┌─────────────────────────────────┐
                      │             main.c              │
                      │  per-rank driver: spawn, drive, │
                      │  flush (Alltoallv), gather, log │
                      └──────────────┬──────────────────┘
                                     │ owns
       ┌──────────────┬──────────────┼───────────────┬───────────────┐
       ▼              ▼              ▼               ▼               ▼
  partition_t     routing_t     scheduler_t /     outbound_t      path_buf_t
  (graph_io)      (routing)     node_scheduler_t  (comm_batch)    (walker)
   TAL per node   peers+time    walker buckets    per-dst queue   completed
       │              │              │               │            paths
       └──────┬───────┘              │               │
              ▼                      │               │
          intmap_t                   ▼               ▼
          (intmap)               walker_t  ──MPI_Alltoallv──► other ranks
                                 (walker)  ◄──MPI_Alltoallv──
```

## Module responsibilities

| Module           | Owns                                          | Depends on             |
| ---------------- | --------------------------------------------- | ---------------------- |
| `config.h`       | constants, walker wire layout, timestamp synth| nothing                |
| `intmap`         | int → int hash table                          | nothing                |
| `chunkio`        | logical-path → on-disk chunk(s) resolution    | nothing                |
| `routing`        | cross-partition routing table (with timestamps)| `intmap`, `chunkio`   |
| `graph_io`       | partition load, TAL build, ID mapping, log I/O| `intmap`, `chunkio`    |
| `walker`         | walker state machine + completed-path buffer  | `graph_io`, `routing`  |
| `scheduler`      | time-bucketed walker scheduler                | `walker`               |
| `node_scheduler` | per-current-node walker scheduler             | `walker`               |
| `comm_batch`     | per-destination outbound buffers for Alltoallv| `walker`               |
| `main.c`         | arg parsing, three run loops, MPI collectives | everything             |

The dependency graph is a DAG. Note **igraph is no longer a dependency** —
the TAL (see below) replaces it, and the only graph "library" call we
needed (connectivity check) was dropped (temporal connectivity has a
different meaning anyway).

## Key data structures

### Walker (wire format)

A walker is exchanged as a flat `int[]`. Layout (see `config.h`):

```
offset  field           meaning
------  ---------       ------------------------------------------
   0    id              globally unique walker id
   1    start_ts        time(NULL) at spawn
   2    end_ts          time(NULL) at completion (0 until then)
   3    hops_out        cross-partition migrations so far
   4    t_cur           temporal cursor; next hop must have t > t_cur
   5..  path nodes      global node ids visited, in order
```

`WALKER_HEADER_INTS = 5`; total capacity `= WALKER_HEADER_INTS + nsteps`.
Dead-ended walkers pad unused path slots with `WALKER_DEAD_END_PAD` (-1)
so the on-disk log stays fixed-width.

In memory the running walker (`walker_t`) carries the wire buffer plus a
cached `cur_local` (local id of the current node) and `cap_ints`. The
cache is **not** transmitted — the receiver re-derives `cur_local` from the
last path entry via `g2l`.

### partition_t (per-rank subgraph + TAL)

```c
typedef struct { int neighbor_local; int t; }   tal_edge_t;   // sorted by t
typedef struct { tal_edge_t* edges; int size; }  tal_t;

struct partition_t {
    tal_t*   tals;     // tals[local_id] = time-sorted adjacency list
    int*     l2g;      // l2g[local] = global
    int      nnodes;
    intmap_t g2l;      // g2l[global] = local
    int      t_min, t_max;
};
```

Each node's `tal_t` is a contiguous array of `(neighbor, t)` sorted by `t`
ascending. The time-respecting next hop is then
`tal_upper_bound(tal, t_cur)` (binary search, O(log deg)) followed by a
uniform draw from the valid suffix.

`partition_ensure_node()` lazily appends a node with an empty TAL — needed
for boundary nodes whose only edges are cross-partition (they appear in the
routing table but not in any local `sub<r>.txt`).

### routing_t (time-aware cross-partition lookup)

```c
typedef struct { int dst_global; int dst_proc; int t; } route_peer_t;  // sorted by t
typedef struct { int src_global; route_peer_t* peers; int npeers; } route_entry_t;

struct routing_t {
    route_entry_t* entries;
    intmap_t       index;   // src_global -> idx into entries
};
```

Peers are sorted by `t` so cross-partition edges are filtered by the same
`routing_upper_bound(re, t_cur)` binary search. A hop samples uniformly
from `(local valid edges) ∪ (remote valid edges)`.

### Timestamp synthesis

`config.h`'s `synth_timestamp(a, b)` returns a deterministic, symmetric
timestamp for an edge when the input file has only 2 columns. 3-column
files (`src dst t`) use the file's value. This lets the static datasets run
unchanged while real temporal datasets (Wikipedia/Reddit/MOOC,
Stack-Overflow) are read directly.

## Walker lifecycle

`walker_spawn` (or `walker_adopt` on receipt) eagerly places the walker on
a node, so every `walker_step` takes exactly one edge:

```
   spawn (random local node)        adopt (from peer, node via g2l)
        │                                │
        ▼                                ▼
   walker_step ◄─── CONTINUE ─────── walker_step
        │
        ├── DONE / DEAD_END ──► walker_finalize ─► path_buf_push
        │
        └── MIGRATE ──► outbound_push ──► (next Alltoallv flush)
```

`walker_step` returns:

- `WALKER_STEP_CONTINUE` — local hop; caller re-schedules the walker.
- `WALKER_STEP_MIGRATE` — next node is on `*out_dst_rank`; caller queues it
  in the outbound buffer for the next flush.
- `WALKER_STEP_DONE` — path reached `nsteps`.
- `WALKER_STEP_DEAD_END` — no valid future edge; terminate early.

## Scheduling policies

Three intra-rank scheduling strategies, selected at runtime
(`delta_t`/`policy` args; see README). All except drive-to-death batch
their cross-rank sends.

| policy | module | bucket key | rationale |
| --- | --- | --- | --- |
| drive-to-death | (none) | — | run each walker to completion before the next; legacy baseline |
| single-bucket | `scheduler` (Δt=0) | all in one bucket | round-robin one step per walker; simplest batching |
| time-window | `scheduler` (Δt>0) | `t_cur / Δt` | group temporally-close walkers (hypothesised cache locality) |
| node-grouping | `node_scheduler` | `cur_local` | group walkers on the same node → share one TAL |

Empirically (results.md): time-window and node-grouping do **not** beat
single-bucket on these workloads — the locality payoff is smaller than the
bucket-bookkeeping cost. They remain in the tree as ablation baselines and
because they may help at workloads with much higher walker density.

## Communication batching (Wave 4)

The dominant cost in the naive design was one blocking `MPI_Send` per
migrating walker. `comm_batch` replaces this:

1. A migrating walker is appended to `outbound[dst_rank]` as
   `[len, walker_data...]`.
2. When the local scheduler drains, `flush_round` performs a single
   `MPI_Alltoall` (exchange counts) + `MPI_Alltoallv` (exchange walker
   bytes) across all ranks.
3. Received chunks are parsed back into `walker_t` and inserted into the
   local scheduler (or finalised if already full).

This turns *N* small blocking sends into one collective per round and is
the source of the 78–347× speedup.

## Main loop (per rank)

```c
seed_rng(rank);
load partition (sub<r>.txt -> TAL) + routing (rt<r>.txt, +timestamps)
ensure routing-source nodes exist in g2l (boundary-only nodes)
spawn nwalkers_per_rank walkers into the scheduler

while (global_done < total_walkers) {
    drain local scheduler:           // process every bucket
        walker_step each walker -> CONTINUE/DONE/DEAD_END/MIGRATE
        MIGRATE -> outbound_push
    flush_round();                   // MPI_Alltoallv exchange + absorb
    MPI_Allreduce(paths.nwalkers -> global_done);
}
MPI_Gatherv all paths to rank 0; write log
```

Subtleties:

- **Termination** still uses an `MPI_Allreduce` of completed-walker counts
  per round. Correct but a known scaling cost; the `flush_round`
  collective also synchronises ranks, so the loop cannot deadlock.
- **Drive-to-death keeps per-walker `MPI_Send`** (it has no natural batch
  point) — this is the naive baseline (config A in results.md).
- **Arrived-already-full walkers** are finalised on receipt without a step.

## Lessons from the original prototype (carried over from 0.2)

The baseline refactor fixed seven concrete issues; all still hold:

1. Per-step O(nnodes) global→local lookup → cached `cur_local` + `g2l` hash.
2. Per-step O(rt_size) routing scan → `routing_lookup` O(1) via `intmap`.
3. O(nnodes²) node densification + temp file → hash-backed single pass.
   (Wave 1 further replaced the igraph build with direct TAL construction.)
4. Per-step walker `realloc` → single allocation of full capacity.
5. Per-walker `paths` realloc → doubling `path_buf_t`.
6. Identical RNG seed across ranks → rank-mixed seed.
7. Hard `exit(0)` on malformed routing line → warn and skip.

## Large-file chunking

Every on-disk text file is kept below `MAX_CHUNK_BYTES` (90 MiB, comfortably
under common 100 MB limits) so no single file trips GitHub / sync-service
size caps.

**Naming convention.** A logical file `foo.txt` is stored either as the
single file `foo.txt` (when small) or as contiguous parts
`foo.txt.part000`, `foo.txt.part001`, … (when large). Splitting always
happens at line boundaries, so no walker / edge / routing record is broken
across parts.

**Reading** (`chunkio.resolve_chunks`). A logical path resolves to:
- `[foo.txt]` if `foo.txt` exists as a regular file, else
- `[foo.txt.part000, foo.txt.part001, …]` (stop at the first missing index).

`partition_load_edgelist` and `routing_load` iterate the resolved list, so
split and unsplit inputs are handled identically. `partition_metis.py`
mirrors this in `resolve_chunks` / `load_edges`.

**Writing.** `log_write` (and the Python `write_split`) stream rows/lines
into `foo.txt.part000`, rolling to the next part whenever adding the next
row would exceed `MAX_CHUNK_BYTES`. If only one part results, it is renamed
back to `foo.txt` (small outputs stay single-file). Before writing, any
stale single file **and** contiguous parts from a prior run with the same
name are removed, so leftover higher parts cannot be misread as current
data.

This applies to: log output (`log/`), partition edge lists and routing
tables (`data/<P>/…sub<r>.txt`, `…rt<r>.txt`), and raw 3-column edge files
(`data/<name>.txt`).

## Configuration and tunables (`config.h`)

| Macro | Purpose |
| --- | --- |
| `WALKER_HEADER_INTS` | header ints in wire format (now 5) |
| `WALKER_TCUR` / `WALKER_*` | wire-layout accessor macros |
| `WALKER_INITIAL_TCUR` | initial cursor (-1, so first hop accepts any edge) |
| `WALKER_DEAD_END_PAD` | sentinel (-1) padding early-terminated walkers |
| `TSYNTH_MAX` + `synth_timestamp` | deterministic timestamp synthesis for 2-col input |
| `DEFAULT_DATASET/NWALKERS/NSTEPS/MODE/DELTA_T` | argv defaults |
| `DATA_DIR` / `LOG_DIR` | I/O roots |
| `TAG_WALKER` | MPI tag for in-flight walkers |
| `MAX_CHUNK_BYTES` | max single-file size (90 MiB) before splitting into `.partNNN` |

## Known limitations / future work

- **Scaling is communication-bound at light per-rank load** (negative strong
  scaling at 200K walkers; positive at 2M — see results.md §9).
- **Termination via per-round Allreduce** — could use Dijkstra-Scholten.
- **Extreme bipartite graphs (e.g. MOOC, 97 items)** make the naive
  per-walker-send baseline pathologically slow (≈100% cross-rank rate);
  batched modes handle them but were not fully benchmarked.
- **No backward / biased (node2vec-style) walks** — uniform forward only.
- **No GPU / alias-method sampler** — out of scope.
- **Downstream task validation** (CTDNE/CAW link prediction) not yet done.

## See also

- [../README.md](../README.md) — build/run instructions
- [../results.md](../results.md) — full benchmark data
- [`config.h`](../config.h) — wire format & defaults
- [../CHANGELOG.md](../CHANGELOG.md) — version history
