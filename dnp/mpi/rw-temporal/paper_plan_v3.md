# Paper Plan v3 — "Zero-Embedding-Communication Distributed Temporal Node Embedding"

> Working title sharpened toward the headline comm result (was "Co-located
> Distributed Temporal Node Embedding"). Subtitle option: "...via Co-located
> Sharding and Shard-Local Negative Sampling".

Chapter Plan + INSIGHT Collection (academic-paper `plan` mode output).
Source of truth for data: `research_plan_v3.md` §10 (all F4 cluster results),
`nomad_differentiation.md` (positioning).

## Configuration (confirmed)
- **Type:** conference paper, HPC / systems-for-ML (empirical systems study).
- **Venue:** mid-tier HPC — IEEE Cluster / ICPP / IPDPS (final pick deferred).
- **Citation:** IEEE. **Length:** ~10–12 pp, 2-col. **Output:** LaTeX.
- **Framing (per v3 plan):** communication-elimination is the headline; temporal
  quality is a real but secondary, eval-gated result.
- **Scoping:** foregrounded — two-stage is *our* same-engine NOMAD-style baseline
  (not real NOMAD); the co-shard/Alltoallv/staleness mechanics overlap NOMAD
  (static, arXiv 2604.09419). Novelty = temporal specialization + zero-embedding-
  comm shard-local-negative design + T4 partitioner.

---

## The spine (what the paper proves, in order)
1. **Fused (co-shard + shard-local negatives) eliminates embedding communication**
   that a NOMAD-style two-stage scheme must pay → +38%→+154% (2.5× at 32 nodes),
   gap *grows* with scale. Clean, monotone, 3-rep. ← THE headline.
2. **Strong scaling:** compute near-ideal (15.6× over 16× procs); overall 3.7–3.8×,
   communication-capped (comm% → 96%).
3. **Weak scaling:** compute flat (ideal), communication-bound (comm 82–89%).
4. **T4 temporal-reachability partitioning** cuts cross-rank rate −5–6 pt
   (deterministic), −12% exchange / 1.20× wall-clock at largest scale; non-monotone.
5. **Quality (single-machine de-risk):** under METIS shards, ρ=0 shard-local
   negatives ≈ global quality; temporal walks beat static on the hard historical-
   negative eval (wikipedia +0.038, tgbl-review +0.031, both 3-seed non-overlapping).

---

## Chapter Plan

### Ch.1 Introduction  (~1.25 pp)
- **Goal:** motivate distributed temporal walk-embedding + state the comm problem.
- **Arc:** temporal graphs are huge and need time-respecting walk embeddings →
  distributing them is gated by *communication*, not compute → prior distributed
  embedding (NOMAD/DistGER) is static and pays remote-embedding exchange → we
  co-locate embeddings with the graph and use shard-local negatives so the
  cross-network embedding traffic is ~0.
- **Contributions (4 bullets):**
  C1 zero-embedding-comm fused design (co-shard + shard-local negatives);
  C2 T4 temporal-reachability partitioner;
  C3 first distributed continuous-time temporal walk-embedding engine + full
     strong/weak scaling on a real A64FX/Tofu cluster (2→32 nodes, 96→1536 ranks);
  C4 eval-gated temporal quality finding (hard-negative benefit).
- **Open Q for user:** lead sentence — capability ("train on graphs too big for
  one node") vs comm-efficiency ("eliminate embedding comm")? v3 plan leans comm.

### Ch.2 Background & Related Work  (~1.25 pp)
- Temporal random walks & CTDNE; skip-gram negative sampling (SGNS).
- Distributed walk-embedding: DistGER (VLDB'23, static), **NOMAD (2026, static,
  MPI)** — credit it for co-shard/owner-computes/Alltoallv/bounded-staleness.
- Temporal GNNs (TGL/DistTGL) — contrast: memory-module, not walk embedding.
- Temporal LP evaluation: Poursafaei/EdgeBank (NeurIPS'22), TGB — sets up why we
  use historical negatives.
- **The empty cell:** distributed × continuous-time × walk-embedding.

### Ch.3 System Design  (~2.5 pp) — the core
- 3.1 Co-shard: embedding table partitioned identically to the graph, so **most
  positive (center, context) pairs are local** (E1 cross-fraction evidence; T4
  lowers it further). The **minority cross-partition positive pairs are dropped**,
  not piggybacked — measured cost ≤2.5 AP (inc-2 deferred). State honestly: the
  migration-piggyback was the original §1 design but is **not implemented**; the
  communication win does **not** come from positive pairs.
  → The embedding-comm savings come almost entirely from **shard-local negatives**
  (3.3) eliminating remote-negative traffic (remote_neg ≈ 18× cross_pairs).
- 3.2 Fused sampling–training: in-walk SGNS in the batched scheduler, no corpus
  materialization; cursor-band locality.
- 3.3 **Shard-local negatives (the novelty):** sample negatives from local shard
  only ⇒ zero remote-negative traffic. Why it works: graph-clustered (METIS/T4)
  shards make local negatives useful *hard* negatives (E2 second correction).
  State honestly: the importance-weight + ρ-exchange machinery was *designed* then
  *dropped* once METIS sharding made ρ=0 ≈ global — report this as a finding.
- 3.4 **T4 temporal-reachability partitioner:** pilot walk pass → empirical
  per-edge traversal weights → METIS. The trivial analytic weight fails; only the
  pilot-based weight works (de-risk result).
- 3.5 Two-stage (NOMAD-style) baseline: remote-embedding Alltoallv exchange — what
  fused avoids. **Explicitly: same engine, our baseline, not real NOMAD.**

### Ch.4 Implementation  (~0.75 pp)
- MPI C, flat one-rank-per-core, A64FX aarch64, `mpifccpx -Kfast`.
- Batched `MPI_Alltoallv` migration; `emb_xchg` phase for two-stage.
- PHASE timing instrumentation (compute/exchange/allreduce/emb_xchg, comm_frac).
- Portability notes (empty-partition guard, portable string/line I/O) — brief.

### Ch.5 Evaluation  (~3.5 pp)
- 5.1 Setup: Wisteria/BDEC-01 Odyssey (A64FX, 48 c/node, ~28 GiB, Tofu-D); datasets
  stackoverflow_a2q (17.8M, scaling); wikipedia/reddit/tgbl-review (quality);
  3-rep medians + ranges.
- 5.2 **F1 fused vs two-stage (HEADLINE):** emb_xchg 0 vs 3.1→6.9 s; +38→154%,
  2.5× at 32 nodes; gap grows. (§10 table.)
- 5.3 **F2 strong scaling:** compute 15.6×/16×; overall 3.7–3.8×; comm%→96%.
- 5.4 **F3 weak scaling:** compute flat; comm-bound 82–89%; cross 59→67%.
- 5.5 **F4 T4 vs static:** cross −5–6 pt deterministic; −12% exch / 1.20× at 1536;
  disclose non-monotone (192 slower).
- 5.6 **F5 quality:** ρ=0 local ≈ global (METIS); temporal>static on historical
  negatives (wikipedia, tgbl-review, non-overlapping seeds); easy negatives
  unreliable. Honest ceiling: static-vector embeddings cap absolute MRR.

### Ch.6 Discussion & Limitations  (~0.75 pp)
- NOMAD overlap stated plainly; novelty = temporal + zero-emb-comm + T4.
- Limitations: single static vector per node caps quality; T4 non-monotone;
  cross-partition positive pairs dropped (inc-2 deferred, ≤2.5 AP).
- **OOM-capability demo CUT** (decision): generator exists but no completed
  billion-edge run; omit to keep the story honest and tight.

### Ch.7 Conclusion  (~0.25 pp)

---

## INSIGHT Collection (load-bearing claims → evidence)

| # | INSIGHT | Evidence (research_plan_v3 §10) | Risk |
|---|---------|----------|------|
| I1 | Fused has zero embedding comm; two-stage pays emb_xchg growing 3.1→6.9 s ⇒ +38→154% | "F4 fused vs two-stage" table | low (clean, 3-rep) |
| I2 | Strong scaling: compute 15.6×/16×, overall 3.7–3.8×, comm%→96% | "F4 strong scaling" table | low |
| I3 | Weak scaling: compute flat, comm-bound 82–89%, cross 59→67% | "F4 weak scaling" table | low |
| I4 | T4 cuts cross −5–6 pt deterministically; 1.20× / −12% exch at 1536; non-monotone | "F4 headline / 3-rep" table | medium (non-monotone — must disclose) |
| I5 | Under METIS shards, ρ=0 local negatives ≈ global (96–97% MRR) | §8 "SECOND CORRECTION" | medium (3–5% reddit gap; single-machine) |
| I6 | Temporal>static on historical negatives: wiki +0.038, tgbl-review +0.031, 3-seed non-overlapping | "TGB clarity check" | medium (2/3 datasets; easy-neg favors static) |
| I7 | Co-shard mechanics overlap NOMAD (static) — novelty is the temporal specialization | nomad_differentiation.md | reviewer-facing; foregrounded by design |

---

## Decisions locked (this session)
- **Title:** sharpened toward comm — "Zero-Embedding-Communication Distributed
  Temporal Node Embedding".
- **OOM demo:** CUT (design point only, no completed run).
- **Quality (F5):** keep as full subsection 5.6 (second leg of the paper).
- **Framing:** communication-elimination headline (per v3 plan ordering).
- **Scoping:** foreground NOMAD overlap; novelty = temporal + zero-emb-comm + T4.

## Open question still pending
1. **Venue** — Cluster vs ICPP vs IPDPS (sets novelty bar + page limit). Deferred
   by author; default working target = IEEE Cluster (lowest risk for the
   same-engine-baseline positioning).
2. **Ch.1 lead hook** — capability ("graphs too big for one node") vs
   comm-efficiency ("eliminate embedding comm"). Plan currently leans comm; can
   finalize at drafting.
