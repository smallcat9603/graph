# Experimental Results — Distributed CTDW Sampler

All timings are `rank=0 elapsed` (MPI_Wtime around the walk loop, excludes
MPI_Init and final gather/log). Medians are over the listed repetitions.
Times in **milliseconds** unless noted.

---

## 1. Environment

| Item | Value |
| --- | --- |
| Host | container `b863903130b0` (single node) |
| Cores | 8 (`nproc=8`) |
| Memory | 7.7 GB total, ~7.1 GB available |
| MPI | Open MPI 4.1.2 |
| Compiler | `mpicc` (gcc backend), `-O3 -Wall -Wextra` |
| Build | igraph dependency removed; pure TAL data structure |

`np <= 8` is true parallel; `np = 16` requires `--oversubscribe`
(2 procs/core) and is reported as *indicative only*.

---

## 2. Datasets

3-column temporal edge lists (`src dst t`), undirected, timestamps
normalised so min = 0. JODIE sets (wikipedia/reddit/mooc) are bipartite
(user/item ID spaces merged). Stack-Overflow from SNAP (sx-stackoverflow-a2q).

| short name | nodes | edges | t range (s) | notes |
| --- | --- | --- | --- | --- |
| wikipedia | 9,227 | 157,474 | 0–2,678,373 | JODIE, bipartite |
| reddit | 10,984 | 672,447 | 0–2,678,390 | JODIE, bipartite |
| mooc | 7,144 | 411,749 | 0–2,572,086 | JODIE (not yet benchmarked) |
| stackoverflow_a2q | 2,464,606 | 17,823,525 | 0–~2.7e8 | SNAP, answers→questions |

---

## 3. Configuration legend

CLI: `./rw <dataset> <nwalkers_per_rank> <nsteps> <mode> <delta_t> <policy>`

| mode | meaning |
| --- | --- |
| 1 | full graph on every rank (no cross-rank communication) |
| 0 | METIS-partitioned, each rank owns a subgraph |

| (delta_t, policy) | scheduler |
| --- | --- |
| (-1, 0) | drive-to-death (one walker to completion at a time) |
| (0, 0) | single-bucket (all walkers, round-robin one step each) |
| (N>0, 0) | time-window (bucket by `t_cur / N`) |
| (0, 1) | node-grouping (bucket by current node) |

Communication: drive-to-death uses per-walker blocking `MPI_Send`;
all batched schedulers use one `MPI_Alltoallv` per round.

Canonical 6-way labels used below:

| label | mode | delta_t | policy | description |
| --- | --- | --- | --- | --- |
| A | 0 | -1 | 0 | partition + drive + per-walker send (naive baseline) |
| B | 0 | 0 | 0 | partition + single-bucket + Alltoallv |
| C | 0 | 5e6 | 0 | partition + time-window + Alltoallv |
| D | 0 | 0 | 1 | partition + node-grouping + Alltoallv |
| E | 1 | -1 | 0 | full-graph + drive (no-communication reference) |
| F | 1 | 0 | 1 | full-graph + node-grouping (reference) |

---

## 4. Algorithm validation — dead-end rate vs walk length

wikipedia (real timestamps), np=1, 200 walkers, mode=1. A walker
"dead-ends" when no future edge (t > t_cur) exists. Confirms the
time-respecting constraint behaves as predicted (rate rises with nsteps).

| nsteps | dead-ended / total | rate |
| --- | --- | --- |
| 3 | 3 / 200 | 1.5% |
| 5 | 46 / 200 | 23% |
| 8 | 127 / 200 | 63% |
| 12 | 184 / 200 | 92% |
| 20 | 200 / 200 | 100% |

Interpretation: under uniform forward sampling, `t_cur` converges
geometrically toward `t_max`; walk length is intrinsically bounded
regardless of dataset. Matches CTDNE-style truncated-walk behaviour.

---

## 5. Main result — 6-way scheduler ablation (np=4, 50K walkers/rank, 30 steps)

5 repetitions; median reported. Raw values in parentheses where notable.

### wikipedia (157K edges)

| label | scheduler | median (ms) |
| --- | --- | --- |
| A | drive + per-walker send | **3350** |
| B | single-bucket + Alltoallv | 46.4 |
| C | time-window + Alltoallv | 40.7 |
| D | node-grouping + Alltoallv | 42.7 |
| E | full-graph drive (ref) | 21.4 |
| F | full-graph node-group (ref) | 41.7 |

### reddit (672K edges)

| label | scheduler | median (ms) |
| --- | --- | --- |
| A | drive + per-walker send | **29,826** (26.5–101.4 s, high variance) |
| B | single-bucket + Alltoallv | 187 |
| C | time-window + Alltoallv | 405 |
| D | node-grouping + Alltoallv | 198 |
| E | full-graph drive (ref) | 101 |
| F | full-graph node-group (ref) | 180 |

### stackoverflow_a2q (17.8M edges)

| label | scheduler | median (ms) |
| --- | --- | --- |
| A | drive + per-walker send | **216,912** (~217 s; 161–331 s) |
| B | single-bucket + Alltoallv | 629 |
| C | time-window + Alltoallv | 625 |
| D | node-grouping + Alltoallv | 788 |
| E | full-graph drive (ref) | 223 |
| F | full-graph node-group (ref) | 392 |

### Speedup of communication batching (A → best batched)

| dataset | naive A | best batched | speedup |
| --- | --- | --- | --- |
| wikipedia | 3350 ms | 40.7 ms (C) | **82×** |
| reddit | 29,826 ms | 187 ms (B) | **160×** |
| stackoverflow_a2q | 216,912 ms | 625 ms (C) | **347×** |

**Speedup grows with graph size.**

### Cross-cutting findings

1. Per-walker `MPI_Send` (A) is unscalable; cost grows with #cross-partition
   hops (78–347× slower than batched).
2. Once communication is batched (B/C/D), scheduler choice contributes
   <20% and varies by dataset. **single-bucket (B) is often the best**, and
   node-grouping (D) can be *worse* at large node counts (2.6M buckets of
   bookkeeping overhead with ~0.08 walkers/bucket).
3. Batched partitioned (B/D) runs within ~2–3× of the memory-prohibitive
   single-rank reference (E), making distributed CTDW practical for graphs
   exceeding single-node memory.

---

## 6. Time-window Δt sensitivity (earlier run, mode=1, np=4, 50K/rank, 30 steps, SO_a2q)

3 reps; median. Shows time-window batching has **no sweet spot** — every
Δt is slower than drive-to-death in the no-communication regime. (This
predates communication batching; included as a negative result.)

| Δt | buckets | median (ms) |
| --- | --- | --- |
| drive-to-death | — | 147 |
| single-bucket | 1 | 298 |
| 100,000 | ~2700 | 394 |
| 1,000,000 | ~270 | 389 |
| 5,000,000 | ~54 | 462 |
| 15,000,000 | ~18 | 451 |
| 50,000,000 | ~6 | 322 |

Finding: no U-shape; bucketing overhead dominates regardless of Δt when
the graph fits in cache and there is no communication to amortise.

---

## 7. Strong scaling — light load (single-bucket B, total = 200K walkers, 30 steps)

3 reps; median. Total work fixed; more ranks ⇒ less work per rank.
Ideal = time halves per doubling. `*` = oversubscribed.

| np | walkers/rank | reddit (ms) | SO_a2q (ms) |
| --- | --- | --- | --- |
| 4 | 50,000 | **80** | **311** |
| 8 | 25,000 | 90 | 428 |
| 16* | 12,500 | 98 | 390 |

**Negative scaling**: adding ranks increases time. Communication-bound at
light per-rank load (per-round Alltoallv overhead grows with P; per-rank
compute too small to amortise).

---

## 8. Weak scaling (single-bucket B, 50K walkers/rank, 30 steps)

3 reps; median. Per-rank work fixed; total grows with np.
Ideal = time flat. `*` = oversubscribed.

| np | total walkers | reddit (ms) | SO_a2q (ms) |
| --- | --- | --- | --- |
| 4 | 200,000 | **88** | **314** |
| 8 | 400,000 | 145 | 708 |
| 16* | 800,000 | 263 | 703 |

Weak-scaling efficiency at np=8: reddit 61%, SO 44%. Time grows with np —
again Alltoallv-bound.

---

## 9. Strong scaling — heavy load (single-bucket B, total = 2M walkers, 30 steps, SO_a2q)

3 reps; median. 10× the light-load total to give each rank enough compute
to amortise communication.

| np | walkers/rank | median (s) | throughput |
| --- | --- | --- | --- |
| 4 | 500,000 | **2.019** | 990K walkers/s |
| 8 | 250,000 | **1.865** | 1.07M walkers/s |

**Positive scaling** (1.08×, 54% efficiency at np=8) — crossover from the
negative scaling seen at 200K. Confirms the system is compute-bound (and
benefits from parallelism) once per-rank workload is large enough.

### Strong-scaling crossover summary (SO_a2q, np=4 → 8)

| total walkers | np=4 | np=8 | scaling |
| --- | --- | --- | --- |
| 200K (light) | 311 ms | 428 ms | 1.4× **slower** |
| 2M (heavy) | 2019 ms | 1865 ms | 1.08× **faster** |

---

## 10. Bottleneck progression (paper narrative)

| stage | suspected bottleneck | finding |
| --- | --- | --- |
| Wave 1–3 | intra-rank scheduling / cache locality | NOT the bottleneck; batching adds overhead with no cache payoff |
| Wave 4 | per-walker MPI_Send | THE bottleneck; Alltoallv batching gives 78–347× |
| Scaling | collective communication (Alltoallv) | dominates at light load; amortised by heavy per-rank compute |

---

## 11. Open / future experiments (not yet run)

- mooc dataset (4th data point)
- clean np=16 on a true 16-core machine (current np=16 is oversubscribed)
- downstream task: feed sampled walks to CTDNE/CAW for temporal link
  prediction (validate walk quality, not just throughput)
- METIS cut-ratio statistics per dataset (printed by partition_metis.py
  but not yet recorded)
- per-rank load-imbalance measurement (walkers completed per rank)
