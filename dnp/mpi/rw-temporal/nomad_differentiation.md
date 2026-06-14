# NOMAD vs. v3 — One-Page Differentiation

NOMAD = arXiv 2604.09419 (Apr 2026), "Generating Embeddings for Massive
Distributed Graphs." The closest prior art to v3's *systems* core. This page
decides what v3 must concede, what survives as novel, and how to position.

## What NOMAD is (read from full text)

- **Model:** LINE 2nd-order proximity; positive pairs from **first-order
  (unbiased) random walks**; negatives from global degree$^{0.75}$.
- **Partitioning:** plain **1D vertex partitioning** (`|V|/p` per rank) —
  *not* edge-cut- or workload-aware.
- **Embeddings:** **owner-computes**, co-located with the vertex partition;
  positive pairs have `owner(u)=p` so **u local, v maybe remote**.
- **Communication:** **2× `MPI_Alltoallv` per batch** — (1) fetch remote
  context + remote-negative embeddings, (2) send gradient deltas back to
  owners; **bounded staleness**, periodic reconcile. **Synchronous within a
  batch** (limited overlap).
- **Scale/eval:** up to 512 Perlmutter nodes, 111M nodes / 6.4B edges, 13
  datasets — **all static**. Baselines: LINE, node2vec, PyTorch-BigGraph,
  GraphVite. Downstream: **node classification only** (Micro/Macro-F1).
- **Not supported:** temporal/dynamic/streaming, GPU, biased (2nd-order) walks.
  No explicit Limitations/Future-Work section.

## Axis-by-axis

| axis | NOMAD (static, 2026) | v3 | status for v3 |
| --- | --- | --- | --- |
| co-shard / owner-computes | ✅ has it | same | **CONCEDE — not novel** |
| batched `Alltoallv` embedding exchange | ✅ 2×/batch | same | **CONCEDE — not novel** |
| bounded staleness | ✅ | same | **CONCEDE — not novel** |
| "positive pairs mostly local" | ✅ stated | same | **CONCEDE — not novel** |
| **time-respecting (continuous-time) walks** | ❌ static, unbiased | ✅ t>t_cur | **NOVEL — core gap** |
| **temporal-reachability partitioning (T4)** | ❌ naive 1D | ✅ METIS + traversal weight (measured −10..−20pt cross) | **NOVEL + beats NOMAD's partitioner** |
| **time-window (cursor-band) locality** | ❌ no time axis | ✅ banded schedule + comm | **NOVEL** |
| **migration-piggybacked positive gradient** | ❌ separate 2× Alltoallv | ✅ gradient rides the walk-migration msg (fused) | **NOVEL (fusion vs NOMAD's 2-phase)** |
| **negative-sample traffic** | fetches **remote** negatives (extra Alltoallv) | shard-local + importance weight (E2: recovers global AP) | **DIFFERENT trade-off — defensible as comm saving over NOMAD** |
| async / overlap | synchronous within batch | banded async overlap | NOVEL-ish (also v2 territory) |
| downstream task | node classification, static | **temporal link prediction** | NOVEL (untouched by NOMAD) |

## NOMAD's weaknesses v3 can exploit

1. **Naive 1D partitioning** — leaves communication on the table; v3's T4 is
   strictly better (and we measured the cross-fraction drop). Easy win.
2. **Remote negatives cost Alltoallv traffic** — v3's shard-local + correction
   cuts it; E2 shows quality is recoverable. **This re-validates T3 as a
   communication optimization over NOMAD** (not just an alternative).
3. **Synchronous within batch** — v3's banded overlap can hide more.
4. **Static only; node-classification only** — the entire temporal axis + the
   natural temporal task (link prediction) are open.

## Repositioned v3 thesis (honest, defensible)

> **"Distributed embedding of continuous-time temporal graphs."** Take
> NOMAD-style co-shard distributed walk embedding as the (static) starting
> point — *we do not claim those mechanics* — and contribute the **temporal
> specialization**: time-respecting walk sampling, **temporal-reachability
> partitioning (T4)** that beats NOMAD's naive 1D, **time-window locality** for
> banded sampling+training+comm, and **temporal-specific communication savings**
> (migration-piggybacked positive gradients + shard-local corrected negatives)
> that cut the remote traffic NOMAD pays. Evaluate on **temporal link
> prediction**, which NOMAD never addresses.

- **Baseline to beat:** NOMAD (run it static / as a temporal-agnostic ablation)
  + v1's two-stage sampler. Show v3 ↓ communication and ↑ temporal-LP quality.
- **Venue realism:** incremental over NOMAD on the systems mechanics, but the
  temporal extension + T4 + temporal-LP is a coherent delta → **mid-tier systems
  venue (Cluster/ICPP/IPDPS)**; SC only if T4 + comm savings + scale are strong.
- **Do NOT** frame co-shard/batched-Alltoallv/bounded-staleness as contributions.

## Baseline strategy (code availability checked 2026-06-14)

- **NOMAD: no public code** (preprint, arXiv 2604.09419, Apr 2026, no venue).
  → Do **not** reproduce its codebase. Implement its mechanics (co-shard,
  owner-computes, 2× batched `Alltoallv`, bounded staleness) as a
  **temporal-agnostic / static ablation config of our own engine**; cite NOMAD
  as the origin. Avoids the "is your reimplementation faithful?" dispute.
- **DistGER: public code** ✅ [github.com/RocmFang/DistGER] (VLDB'23, C++/MPI,
  builds on KnightKing). → Use as the **released static SOTA baseline**: run it
  treating the temporal graph as static, and show v3's temporal-aware version
  wins on temporal-LP quality and/or communication ("static SOTA cannot exploit
  time; we do").
- **Three-tier baseline:** DistGER (released, static) ⟶ our NOMAD-style static
  ablation ⟶ v3 temporal. Plus v1 two-stage to ablate fusion. The
  "can-we-benchmark" risk is resolved.
- Optional: DistGER's codebase is MPI C++ — could also be inspected for
  partitioner/skip-gram ideas, but v1's engine is already temporal-native, so
  build on v1, not DistGER.

## Risk that remains

- NOMAD is 2 months old (preprint); a temporal follow-on could appear before
  submission — re-check near submission.
- The delta is "extension," not "new paradigm." If the temporal pieces don't
  show a clear win (comm and quality) over a temporal-agnostic NOMAD baseline,
  the story is thin. T4's measured cross-fraction reduction is the strongest
  single piece of evidence in hand; lead with it.
