# Paper Plan — Distributed Continuous-Time Random Walk Sampler

Produced via `/ars-plan` (academic-paper plan mode, Socratic chapter-by-chapter).
Target: generic mid-tier systems conference, ~10 pages double-column.
Primary narrative: systems contribution; empirical study supporting.

This is the single input for a later `/ars-full` or `/ars-outline` run.

---

## INSIGHT Collection

### INSIGHT: thesis_statement

> In distributed continuous-time random walks (CTDW), the dominant
> performance bottleneck is **cross-rank communication granularity, not
> intra-rank scheduling**. Batching walker migrations into a collective
> `MPI_Alltoallv` per round yields **78–347× speedup** over per-walker
> `MPI_Send` (speedup growing with graph size) and makes distributed CTDW
> practical; the choice of intra-rank scheduling policy contributes <20%.

Reader take-away: do not port static-graph batching/locality intuitions
to temporal distributed walks — fix communication first.

### INSIGHT: bottleneck_progression

Three bottlenecks revealed in sequence (the paper's empirical spine):
1. Intra-rank scheduling / cache locality — hypothesised, but NOT the
   bottleneck (batching adds overhead with no cache payoff because
   time-similar walkers sit on different nodes → time-similarity ≠
   address-similarity).
2. Per-walker `MPI_Send` — THE bottleneck; collective batching gives
   78–347×.
3. Collective communication (`MPI_Alltoallv` + per-round Allreduce) —
   dominates at light per-rank load; amortised by heavy per-rank compute
   (strong-scaling crossover).

### INSIGHT: argument_weaknesses (must pre-empt in Discussion/Limitations)

| weakness | reviewer attack | defence |
| --- | --- | --- |
| doesn't beat single-rank | "mode=1 21ms < best mode=0 43ms, why distribute?" | capability scaling: only option when graph exceeds rank memory; within 2× |
| speedup vs a weak baseline | "per-walker send is a strawman" | it is the actual legacy implementation; reveals why naive porting fails |
| scaling only to np=8 | "8-core single node isn't distributed scaling" | np=16 marked oversubscribed; weak + strong-crossover show the trend; multi-node = future work |

---

## Chapter Plan (7 chapters, ~10 pages)

### 1. Introduction (~1.5 pp)

- **Urgency**: temporal GNNs (TGN, CAW, CTDNE) depend on time-respecting
  walk sampling; temporal graphs outgrow single-node memory; no open-source
  distributed CTDW sampler exists.
- **Gap (one sentence)**: the "distributed × temporal" cell is empty —
  KnightKing is static-distributed, TGL/TGOpt are temporal-single-machine.
- **Contributions**:
  1. First open-source MPI distributed CTDW sampler.
  2. Communication batching (Alltoallv) → 78–347× over per-walker send,
     speedup growing with graph size.
  3. Counter-intuitive finding: scheduling policy contributes <20%; the
     bottleneck is communication.
  4. Complete design-space ablation + strong/weak scaling characterisation.

### 2. Background & Related Work (~1.5 pp)

- Four threads: (1) static distributed walks [KnightKing, ThunderRW,
  FlashMob]; (2) temporal walks, single-machine [TGL, TGOpt, CAW, CTDNE];
  (3) graph partitioning [METIS]; (4) MPI collective communication.
- Leads to: nobody combines distributed + temporal; the static-graph
  batching intuition does not transfer naively.
- Disagreement: static walk systems implicitly assume locality/scheduling
  drives walk performance — we show it does not in the temporal-distributed
  setting.
- Key figure: 2×2 positioning table (static/temporal × single/distributed);
  we fill the bottom-right cell.

### 3. System Design (~2.5 pp, core)

- 3.1 Temporal graph model + TAL (time-sorted adjacency, binary-search
  `upper_bound(t_cur)`).
- 3.2 Temporal walker (`t_cur`, time-respecting sampling from local TAL ∪
  remote routing peers with t > t_cur, dead-end termination).
- 3.3 Partitioning + routing table (METIS; cross-partition edges carry
  timestamps).
- 3.4 Scheduling policies (drive-to-death / single-bucket / time-window /
  node-grouping).
- 3.5 **Communication batching via MPI_Alltoallv (key subsection)** —
  per-destination outbound buffers, one collective per round vs N blocking
  sends.

### 4. Implementation (~1 pp)

- ~3000 lines C + MPI; igraph dropped in favour of in-house TAL.
- Key structures: intmap hash, walker wire format, `[len, data]` Alltoallv
  packing, `partition_metis.py`.
- Open-source link. Keep short to avoid overlap with §3.

### 5. Evaluation (~3 pp, heaviest)

- 5.1 Setup: 4 datasets (wikipedia 157K, reddit 672K, stackoverflow_a2q
  17.8M edges; facebook/static optional), 8-core Open MPI 4.1.2 node,
  5-rep median protocol.
- 5.2 Algorithm validation: dead-end rate vs nsteps (1.5%→100%), confirms
  time-respecting constraint.
- 5.3 **Main result**: 6-way ablation × 3 datasets → 78×/160×/347×.
- 5.4 Scheduler comparison: single-bucket often best; node-grouping worst
  at large node counts (the surprising finding).
- 5.5 Scaling: strong (light = negative, heavy 2M = positive crossover
  1.08× @ np=8), weak (44–61% efficiency).
- 5.6 Δt sensitivity: no sweet spot (negative result).
- Figures: log-axis bar chart; speedup-vs-size; strong-scaling crossover;
  Δt sensitivity.
- Data source: `results.md`.

### 6. Discussion / Limitations (~1 pp)

- Take-away: fix communication before scheduling in temporal distributed
  walks.
- Practice: distribute only when the graph exceeds single-node memory;
  don't add ranks at light load.
- Limitations (stated up front): communication-bound scaling; per-round
  Allreduce termination; extreme-bipartite pathology (MOOC); no
  backward/biased walks; no GPU; downstream task not yet validated.

### 7. Conclusion (~0.5 pp)

- First distributed CTDW sampler; communication batching is the key;
  empirical bottleneck-progression story.
- Future: downstream CTDNE/CAW link-prediction validation, streaming edge
  ingestion, backward walks, GPU-MPI hybrid.

---

## Evidence map (chapter → results.md section)

| chapter | evidence |
| --- | --- |
| §1 contributions | results.md §5 (speedups), §9 (scaling) |
| §3.1–3.5 design | ARCHITECTURE.md (data structures, schedulers, comm batching) |
| §5.2 validation | results.md §4 (dead-end vs nsteps) |
| §5.3 main result | results.md §5 (6-way ablation, 3 datasets) |
| §5.4 scheduler | results.md §5 cross-cutting findings + §6 (Δt sweep) |
| §5.5 scaling | results.md §7 (strong light), §8 (weak), §9 (strong heavy) |
| §6 limitations | results.md §11 + ARCHITECTURE.md "Known limitations" |

## Related-work citations to gather (during writing)

- KnightKing (SOSP'19), ThunderRW (VLDB'21), FlashMob (PPoPP'23)
- TGL (NeurIPS'22), TGOpt (PPoPP'23), CAW (ICLR'21), CTDNE (WWW'18 companion)
- METIS (Karypis & Kumar), DistDGL (IA3'20 / VLDB'21)
- JODIE (KDD'19) + SNAP sx-stackoverflow for datasets

## Next steps

1. (optional) `/ars-outline` — expand this plan into a detailed
   per-paragraph outline with evidence anchors.
2. `/ars-full` — generate the full draft from this plan.
3. Gather/verify the related-work citations above before drafting §2.
4. `visualization_agent` (in full mode) — generate the 4 figures from
   results.md data.
