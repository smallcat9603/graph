# Paper Plan — Distributed Continuous-Time Random Walk Sampler

Produced via `/ars-plan` (academic-paper plan mode, Socratic chapter-by-chapter).
Target: NBiS 2026, Springer svproc, 12 pages single-column.
Primary narrative: an honest characterisation study of where the cost of
distributed temporal walk sampling goes; systems contribution supporting.

> **Reframed 2026-05-29 after the `/ars-reviewer` pass.** The thesis was
> recast from a speed claim ("we are 78–347× faster") into a characterisation
> / batching-necessity framing, and the single-node scope is now disclosed
> up front rather than defended. This file is synced to the current
> `author.tex`; the rationale for each change is in `RESPONSE.md`.

---

## INSIGHT Collection

### INSIGHT: thesis_statement

> We build the first distributed CTDW sampler and use it as a measurement
> instrument to ask where the cost of distributed temporal walk sampling
> actually goes. The answer is **cross-rank communication granularity, not
> intra-rank scheduling**: migrating each walker in its own blocking message
> is catastrophic, whereas batching a round's migrations into one collective
> `MPI_Alltoallv` is **78–347× faster** (growing with graph size). Once
> migration is batched, the scheduling policy contributes <20%. We treat the
> per-walker-send design as the naive reference point, not a competing
> system; the contribution is the characterisation, not a speed record.

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

### INSIGHT: argument_weaknesses (pre-empted by honest scoping, not defended away)

| weakness | reviewer attack | how the paper handles it |
| --- | --- | --- |
| doesn't beat single-rank | "mode=1 21ms < best mode=0 43ms, why distribute?" | conceded: our graphs fit one node, so this is a design characterisation, not a capability demonstration; value is capability only beyond single-node memory (§5.1, §6) |
| speedup vs a weak baseline | "per-walker send is a strawman" | reframed: per-walker send is the naive reference point, not a competitor; the contribution is "batching is necessary," not a speed record (Abstract, §1) |
| single-node, scaling only to np=8 | "8-core single node isn't distributed scaling" | disclosed up front: intra-node MPI, no real network; batching gap is a conservative lower bound; multi-node validation = future work (§1 scope ¶, §5.1) |
| scheduling finding may not generalise | DA: "true in cache-resident/short-walk regime only" | scoped explicitly in §6; node-grouping slowdown attributed to an implementation artefact, not a law |

These map to the point-by-point record in `RESPONSE.md`. Three gaps are
acknowledged as needing real experiments (multi-node run, a real-system /
TEA baseline, a downstream link-prediction validation).

---

## Chapter Plan (7 chapters, ~10 pages)

### 1. Introduction (~1.5 pp)

- **Urgency**: temporal GNNs (TGN, CAW, CTDNE) depend on time-respecting
  walk sampling; temporal graphs outgrow single-node memory; no open-source
  distributed CTDW sampler exists.
- **Gap (one sentence)**: the "distributed × temporal" cell is empty —
  KnightKing is static-distributed; TEA and TGL are temporal-single-machine.
- **Contributions** (3, matching `author.tex`):
  1. The first open-source MPI engine for distributed sampling of
     continuous-time temporal random walks on vertex-partitioned graphs.
  2. The finding that communication granularity, not scheduling, is the
     dominant factor: collective batching is 78–347× faster than per-walker
     send (growing with graph size), while the scheduling policy contributes
     <20% once communication is batched.
  3. A scaling characterisation: a crossover from communication-bound at
     light per-rank load to compute-bound (positively scaling) at heavy load.
  Scope is stated up front: single-node intra-node MPI, per-walker send as
  the naive reference point, multi-node validation as future work.

### 2. Background & Related Work (~1.5 pp)

- Four subsections (as written in author.tex §2): (1) walks for
  representation learning [DeepWalk, node2vec; temporal: CTDNE, CAW, TGAT,
  TGN]; (2) walk engines [KnightKing, ThunderRW, FlashMob, GraphWalker];
  (3) temporal and distributed systems [TEA single-machine, TGL, DistTGL,
  DistDGL]; (4) partitioning and communication [METIS, MPI].
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

### 5. Evaluation (heaviest section) — 5 subsections as written

- 5.1 Setup: 3 datasets (wikipedia 157K, reddit 672K, stackoverflow_a2q
  17.8M edges; mooc dropped — naive baseline does not converge on its
  extreme bipartite structure), 8-core Open MPI 4.1.2 node, 5-rep median
  protocol; single-node intra-node-MPI scope + variance disclosed here.
- 5.2 Walk-length behaviour: dead-end rate vs nsteps (1.5%→100%), confirms
  the time-respecting constraint and bounds per-walker work.
- 5.3 **Communication batching is the dominant factor** (main result):
  6-way ablation × 3 datasets → 78×/160×/347×; plus speedup-vs-graph-size
  figure.
- 5.4 **Scheduling policy is secondary**: single-bucket often best;
  node-grouping worst at large node counts; Δt sweep folded in here (no
  sweet spot, negative result).
- 5.5 Scaling: strong (light = negative, heavy 2M = positive crossover
  1.08× @ np=8), weak (44–61% efficiency).
- Figures (7 total: 6 data figures + 1 pseudocode): round-loop pseudocode
  (in §4); dead-end curve; ablation bar chart (log); speedup-vs-size; Δt
  sweep; strong-scaling crossover; weak scaling. The 6 data figures are B&W
  (hatch / line-style differentiation).
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

## Related-work citations — verified in `author.tex` (source of truth)

All 19 references in `author.tex` had their venue/year/pages/authors
WebSearch-verified on 2026-05-29. This pre-writing wish-list contained
several wrong guesses that were corrected during writing; recorded here so
the corrections are not re-introduced:

- FlashMob is **SOSP'21** (not PPoPP'23); TGL is **PVLDB 15(8) 2022**
  (not NeurIPS'22); ThunderRW is **PVLDB 14(11) 2021**.
- TGOpt was **not** used; the temporal-method set is CTDNE (WWW Companion'18),
  CAW (ICLR'21), TGAT (ICLR'20), TGN (arXiv'20).
- Added during writing: TEA (EuroSys'23, the closest single-machine temporal
  walk engine), GraphWalker (ATC'20), DistTGL (SC'23).
- Unchanged and correct: KnightKing (SOSP'19), METIS (SIAM JSC'98),
  DistDGL (IA3'20), JODIE (KDD'19), SNAP, MPI 4.0, Holme & Saramäki (2012),
  DeepWalk (KDD'14), node2vec (KDD'16).

Refer to the `author.tex` bibliography, not this list, for exact entries.

## Status (as of 2026-05-29)

The full draft is **already written** in the NBiS workspace
(`.../NBiS2026-rw-temporal/LaTeX/Sources/author.tex`), not pending
generation. Done: 12-page svproc draft; 19 citations all WebSearch-verified;
6 B&W figures (`make_figs.py` + `Fig/*.pdf`); simulated 5-reviewer pass with
the response record in `RESPONSE.md`.

**Do NOT regenerate from this plan with `/ars-full`** — it would revert the
reviewer-hardened framing. `author.tex` is the source of truth; this file is
the synced plan-of-record.

Open items needing real experiments (see `RESPONSE.md`): multi-node run,
a single-machine (TEA) baseline, a downstream link-prediction validation.
