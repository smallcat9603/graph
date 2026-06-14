# Research Plan v3 — Co-located Distributed Temporal Node Embedding

Target: a systems-for-ML venue (MLSys / KDD applied / VLDB; fallback Cluster, SC ML track).
Builds on the accepted NBiS 2026 paper (the "v1" characterization study).
Sibling to `research_plan_v2.md` — v2 stays a pure HPC-systems paper; **v3 pivots
toward node embedding** to reach ML venues and demonstrate the out-of-memory
capability v1 never showed.

> **中文导航(这份文件是什么)**
> NBiS 那篇证明了:**通信粒度主导**分布式时序游走成本,批处理迁移快 78–347×,
> 但**始终没演示"超出单机内存的能力"**。v3 用 node embedding 把这个能力做实:
> 把每个节点的嵌入向量**和它的图分区存在同一台机器**(co-shard),让"走子搬家
> 消息"和"向量更新"合二为一 → 跨网络流量≈0;负样本就近挑 + 权重修偏差。
> 目标 venue 从 SC/IPDPS(v2)扩到 MLSys / KDD / VLDB。
>
> **未核查警告**:本计划的 HPC 机制经过 deep-research,但**嵌入这条线的前沿尚未
> 核查**。§7 标了哪些 prior art 已验证、哪些动手前必须查。

---

## 0. The one-sentence thesis

**Store each node's embedding vector on the same machine that owns that node's
graph partition (co-shard), and fuse walk sampling with skip-gram training. Then
the partition that minimizes walker migration also minimizes embedding-update
traffic, because a walk's cross-rank step and its cross-rank embedding update are
the *same event* — and we piggyback the update on the migration message that was
being sent anyway.**

Pay-off: train node embeddings on temporal graphs **too large for one machine** —
the capability v1 motivated but never demonstrated — and report competitive
temporal link prediction (TGB).

---

## 1. The two innovations, by tiny example

### Innovation 1 — co-shard makes embedding updates ride the migration message

Setup: machine **A** owns nodes {1,2,3}, **B** owns {4,5,6}. Each node's vector
lives on its owner (innovation 1). Skip-gram nudges the vectors of **adjacent**
nodes in a walk closer; each pair needs read+write of both vectors.

Walk `1 → 2 → 3 → 4 → 5` produces 4 adjacent pairs:

| pair | location | network? |
| --- | --- | --- |
| (1,2) | A,A | local, free |
| (2,3) | A,A | local, free |
| (3,4) | A,B | **crosses** — and this is exactly the step where the walker migrates A→B |
| (4,5) | B,B | local, free |

The only cross-rank pair coincides with the walker's migration, so its gradient
**rides the migration message**. Network cost for embedding updates:

| | network ops for 4 pairs |
| --- | --- |
| naive (embeddings on separate parameter servers) | **4** |
| co-shard + piggyback | **0 extra** (3 local + 1 piggybacked) |

This is the v1 finding reused: the same partition (and the same batched
`MPI_Alltoallv`) that minimizes migration now also carries the embedding work.

### Innovation 2 — shard-local negative sampling + importance weight

Negative sampling also pushes each node away from random *unrelated* nodes,
drawn from a global noise distribution $P_{\text{global}}$ (≈ visit frequency).
Example $P_{\text{global}}$ over the 6 nodes:

| node | 1 | 2 | 3 | 4 | 5 | 6 |
| --- | --- | --- | --- | --- | --- | --- |
| $P_{\text{global}}$ | 0.10 | 0.15 | 0.15 | 0.20 | 0.20 | 0.20 |

A's mass = 0.40, B's = 0.60. To avoid remote fetches, A samples negatives **only
from {1,2,3}**, renormalized: $P_{\text{local}}(1)=0.25,\ (2)=(3)=0.375$. That
over-samples A's own nodes (1 should be 0.10, A picks it at 0.25) → biased.

**Fix:** weight each sampled negative by
$$w(n) = \frac{P_{\text{global}}(n)}{P_{\text{local}}(n)}.$$
Here $w(1)=0.10/0.25=0.40$, $w(2)=w(3)=0.40$ — all equal to **A's global mass
0.40**. Clean interpretation: *multiply every local-negative update by the
fraction of the world this machine owns.*

Check: $\sum_{n\in A}P_{\text{local}}(n)\,w(n)\,f(n)=0.10f_1+0.15f_2+0.15f_3$ =
exactly the A-part of the true global negative loss. B does {4,5,6} with weight
0.60; **summed across machines, 0.40 + 0.60 = 1.0 = exact global negative
sampling.**

**The research question (the valuable part):** each anchor still needs the
*other* machines' negative pushes (node 1 should also be pushed from 4,5,6 on B).
B sends these back in a **periodic, batched** negative-gradient exchange (on the
existing Alltoallv). How often? Frequent → accurate but more traffic; rare →
cheaper but staler. This **bias–variance / exchange-frequency analysis** is the
core ML-systems contribution.

---

## 2. Positioning — what each related system does NOT do

| system | scope | why it does not cover us |
| --- | --- | --- |
| DistGER (VLDB'23) | distributed RW → embedding, **static** | no timestamps; two-stage (sample then train); embeddings not co-sharded for temporal walks |
| TGL / DistTGL | distributed temporal **GNN training** | memory module, not walk-based embedding |
| SPEED (2023) | streaming **temporal** partition for training | GPU + replication objective, not walk/embedding co-shard |
| TEA (EuroSys'23) | temporal RW, **single machine** | not distributed; no embedding |
| CTDNE | temporal walk → embedding, **single machine, batch** | offline, not distributed, not co-sharded |

**Empty cell = ours:** distributed, continuous-time, co-located graph+embedding,
fused sampling–training.

---

## 3. Contributions (three)

**C1. Co-located graph + embedding sharding with migration-piggybacked updates.**
The partition that minimizes walker migration also minimizes embedding-update
traffic; cross-rank positive-pair gradients ride the migration message. Fused
sampling–training (no walk materialization).

**C2. Shard-local negative sampling with importance-weight correction + a
periodic batched cross-shard negative-gradient exchange**, with a bias–variance
analysis of the exchange frequency. (The genuinely novel, analyzable piece.)

**C3. Out-of-memory capability + scaling.** Train embeddings on a temporal graph
that OOMs a single machine; competitive TGB link prediction; strong/weak scaling
of the three-way (sample/train/communicate) overlap.

---

## 4. Technical thrusts (tied to the existing codebase)

### T1 — Co-shard + extended walker wire format
- `partition_metis.py` additionally assigns each rank the embedding shard
  $E_r=\{e_v:\text{owner}(v)=r\}$ (random init).
- Extend `walker.{c,h}` wire format with the last $w$ path nodes (sliding
  window) so the receiving rank can immediately form boundary (center,context)
  pairs. Cost: ~35 → ~35+w ints (w=5 → +14%); still small, batching unchanged.

### T2 — Fused banded training (innovation 1)
- Each step's context pairs trigger an immediate **local** skip-gram SGD update;
  no corpus on disk.
- Advance by **cursor bands** (reuse `scheduler.{c,h}` as the primary model):
  process all walkers with $t_{\mathrm{cur}}\in[\tau,\tau+\Delta)$ → hot-embedding
  working set stays small and cache-coherent.
- Three-way overlap: `MPI_Ialltoallv` band-$k$ migration+gradients while training
  band-$k$ pairs and stepping band-$(k{+}1)$.

### T3 — Shard-local negatives + correction (innovation 2, the contribution)
- Sample negatives from local shard only; weight each by
  $w=P_{\text{global}}/P_{\text{local}}$ = shard global mass.
- Maintain a small global frequency table, synced once per epoch (one Allreduce).
- **Periodic batched cross-shard negative-gradient exchange** for anchors; tune
  the period; analyze bias–variance vs traffic.

### T4 — Co-design partitioner (serves both traffics)
- Reuse v2's temporal-reachability-weighted partition; the objective now minimizes
  (migration + boundary-pair gradient) jointly — they coincide, so one objective.
- Report: static cut vs temporal-weighted cut → migration AND embedding traffic
  drop together.

### T5 — Async embedding consistency (Hogwild!, temporal version)
- Lock-free async updates; characterize the temporal-walk access sparsity
  (dead-end bounds per-node touches; time-window confines concurrent conflicts to
  one slice) → bounded staleness, convergence argument. Fall back to SSP
  (stale-synchronous) if needed.

---

## 5. Evaluation

| item | plan |
| --- | --- |
| cluster | real multi-node, 4 → 32 physical nodes |
| datasets | TGB (tgbl-comment/flight), full Stack Overflow, GDELT; + synthetic temporal graph that OOMs one node |
| ML metric | **temporal link prediction AP / MRR** (embedding quality) |
| sys metrics | walks·pairs/sec, cross-rank traffic (migration + embedding), convergence steps, exchange-frequency sweep |
| baselines | **DistGER** (distributed walk→embedding, static — key contrast), single-machine CTDNE, TEA→single-node embedding, v1 two-stage (sample-then-train; ablates the fusion win) |
| ablations | T2 fusion on/off; T3 local vs global negatives; T4 static vs temporal partition; T3 exchange period sweep |
| headline figures | (a) two-stage vs fused throughput; (b) migration & embedding traffic drop **together** under temporal partition; (c) corrected local-negative AP matches global-negative AP |

---

## 6. Risks & mitigations

| risk | mitigation |
| --- | --- |
| shard-local negatives lose accuracy | importance weight + periodic global-frequency sync + cross-shard exchange; report AP gap vs global |
| Hogwild staleness hurts convergence | time-window confines conflicts; bounded-staleness on boundary embeddings; SSP fallback |
| "this is just DistGER + timestamps" | differentiate hard: temporal/time-respecting + **co-shard** + **fusion** + migration-piggyback; DistGER is static, two-stage, not co-sharded |
| embedding table itself exceeds memory | co-shard already splits it per rank; quantize/low-precision vectors at extreme scale |
| dead-end → short walks → few pairs | per-step throughput; recency-biased sampling to lengthen effective walks |
| TGB too small to OOM one node | synthetic billion-edge temporal graph |

---

## 7. Prior-art status — Phase 0 lightweight check done (2026-06-14)

**Verified static / different-objective (from v2 deep-research):**
- DistGER (VLDB'23) — distributed RW→embedding, **static**; improves *distributed
  skip-gram* with a "hotness block-based synchronization" to sync node vectors and
  a RW-aware partitioner (−45% cross-machine comm). Static only.
- TGL/DistTGL — distributed temporal **GNN training**, not walk embedding.
- SPEED (2023) — streaming temporal partition, replication objective.
- TEA (EuroSys'23) — single-machine temporal RW.

**⚠ NEW, CLOSE THREAT — NOMAD (arXiv 2604.09419, Apr 2026).** A distributed-memory
**MPI** node-embedding framework that **already does most of v3's *systems* core**,
for **static** graphs:
- samples positive pairs with `owner(u)=p` so **u is always local, v may be
  remote** — *exactly* our "co-shard makes most positive pairs local" observation;
- **owner-computes** embeddings; **bounded staleness** for remote context vectors;
- **batches remote embedding exchange with `MPI_Alltoallv`** — our batching idea;
- negatives "sampled independently of partition boundaries", buffered + batched.

**Implication: the co-shard + owner-computes + batched-`Alltoallv` + bounded-
staleness mechanisms are NO LONGER novel — NOMAD published them (static, 2026).**
What still appears genuinely unoccupied:
- **Distributed *continuous-time temporal* (time-respecting) walk embedding** —
  NOMAD/DistGER/ScaleRunner/KnightKing are all static; CTDNE/WalkingTime/dynamic-
  node2vec/FILDNE are single-machine; TARIS is distributed time-respecting *graph
  processing*, not embedding. The distributed×temporal×walk-embedding cell is open.
- **Temporal-reachability partitioning (T4)** — validated to work here, and not in
  any of the above.
- The **time-window (cursor-band) locality** exploited for scheduling + comm.

**Honest verdict: v3 is now "the temporal specialization of NOMAD-style distributed
walk embedding," not a new paradigm.** It is still publishable IF positioned as
exactly that — NOMAD(static) is the baseline to extend, and the temporal pieces
(time-respecting walks, T4, time-window locality) carry the novelty. The generic
co-shard/communication story must NOT be claimed as new.

**This also dents C2/T3:** NOMAD just *batches global negatives* over `Alltoallv`
and avoids the bias entirely. So shard-local-negatives + importance-weight is an
*alternative* (less traffic, bounded bias), **not an obvious win** — we must
justify it against NOMAD's batched-global-negatives or adopt the latter.

---

## 8. Phase 0–1: De-risking before the big build

**Do not collect data the way v1 did.** v1 already had a working system, so the
contribution *was* the measurement. v3 requires months of new systems+ML
engineering (co-shard, fused training, negative correction, async consistency)
**and its core claims can fail outright** (does co-shard actually localize
updates? does shard-local negative sampling keep accuracy?) **and the novelty may
be pre-empted** (§7). So front-load cheap *falsification* spikes; commit to the
big build only if they pass.

### Phase 0 — kill the novelty risk (days)
Resolve the three §7 threats first. If a distributed streaming/co-shard temporal
embedding system already exists, **pivot now** and save months. Go/no-go gate.

### Phase 1 — two feasibility micro-experiments (1–2 weeks, reuse existing code)

**E1 — premise of Innovation 1 (≈ zero new code).**
Instrument the **existing v1 sampler** to log, over sampled walks, the fraction
of adjacent (center, context) pairs whose endpoints sit on **different ranks**.
- **Pass:** fraction ≈ crossing rate and small (~5–15%) ⇒ most embedding updates
  are local; co-shard + piggyback is justified.
- **Fail:** fraction large ⇒ co-shard saves little; rethink the thesis.
- Runnable **now** on the current codebase — no embedding code needed.

**E1 — results (2026-06-13, static-METIS partition, 500 walkers/rank × 30
steps, forward uniform).** Cross-rank fraction of skip-gram (center, context)
pairs, at training window $w=1,5,10$ (the always-on window-1 counter and the
gated `E1_WINDOW` post-process agree exactly on $w=1$ — cross-validated):

| dataset | $w{=}1$ | $w{=}5$ | $w{=}10$ |
| --- | --- | --- | --- |
| wikipedia np=4 | 11% | 20% | 24% |
| wikipedia np=16 | 17% | 29% | 34% |
| reddit np=4 | 14% | 22% | 25% |
| reddit np=16 | 31% | 41% | 45% |
| **stackoverflow np=4** | 34% | **50%** | 54% |
| **stackoverflow np=8** | 40% | **59%** | 63% |
| **stackoverflow np=16** | 46% | **66%** | 70% |

(static social sanity check, $w{=}1$: facebook 6.8%/15.4%, twitch 36%/49%.)

**Verdict (sharpened — this is a serious warning, not a green light).**
The co-shard premise is partition-, window-, and scale-dependent, and it is
WEAKEST exactly where it matters most:
- Real skip-gram windows ($w=5$–$10$, the DeepWalk/node2vec default) roughly
  **double** the cross fraction vs the window-1 proxy — so the earlier
  window-1 numbers were a large underestimate.
- On the **small** graph (wikipedia) it stays tolerable (~20–34%).
- On the **large** graph (stackoverflow, 17.8M edges — the very out-of-memory
  regime that motivates distribution), a realistic $w=5$ already crosses
  **50–66%** of pairs. **More than half the embedding updates are remote.**
  Under stock METIS, the central "most updates are local" premise of
  Innovation 1 **does not hold on the hardest, most important case.**

**PLAN IMPACT (acted on this finding):**
1. **Co-shard alone is not a sufficient thesis for large temporal graphs.**
   Its headline benefit is *entirely gated* on a partitioner that drives the
   cross fraction down at a realistic window — so **T4 is promoted to THE
   central contribution, not a supporting one**. Quantified, much harder
   target: **cut stackoverflow's $w=5$ cross fraction from ~50–66% toward
   ~20%.** If T4 cannot, the co-shard story collapses on big graphs.
2. **Go/no-go reframed:** before the big build, prototype T4 (temporal-
   reachability-weighted partition) and re-run E1 at $w=5,10$. Decision gate:
   T4 must show a large cross-fraction drop on stackoverflow. If it cannot,
   **pivot** — e.g., to the v2 pure-HPC plan, or to embedding-replication /
   bounded-staleness designs that tolerate high cross rates instead of
   assuming locality.
3. Any "co-shard saves ~all embedding traffic" claim must be stated as
   **conditional on (graph, window, partition quality)** with this E1 table as
   the evidence — never as a universal property.

> Reproduce: `E1_WINDOW=10 mpirun --oversubscribe -np <P> ./rw <dataset> 500 30 0`
> (window-1 prints by default; `E1_WINDOW=w` adds the $1..w$ curve).

**T4 prototype — go/no-go on the partitioner (2026-06-13, np=4).** Two edge
weightings were tested against the unweighted baseline, measured by E1.

E1 cross-rank pair fraction at $w{=}5$, np=4 (walk-length median in parens):

| dataset (walk median) | baseline | analytic "earliness" | **empirical T4** |
| --- | --- | --- | --- |
| wikipedia (6) | 18.95% | 22.45% ❌ worse | **15.86%** (−16%) |
| stackoverflow (5) | 50.07% | — | **39.68%** (−21%) |
| mooc (13, time-dense) | 47.05% | — | **26.64%** (−43%) |

**Findings:**
1. **The cheap analytic weight FAILS** — weighting edges by earliness in time
   slightly *increased* the cross fraction at every scale tried. A trivial
   temporal weight is not enough; T4 needs real traversal statistics.
2. **The empirical weight WORKS** — a pilot pass that simulates walks and
   counts per-edge traversals, fed to METIS as `eweights`, reduces the cross
   fraction on both graphs. T4 is **viable, but only in its principled
   (pilot-based) form** — a genuine build, not a one-line heuristic.
3. **But even with empirical T4, the large graph still crosses ~40% at
   $w=5$** (down from 50%, not down to wikipedia's ~16%). Co-shard's
   "most updates are local" premise stays only *partially* satisfied on the
   hardest, most important case.
4. **Walk-triviality concern RETRACTED (it was a misread metric).** The
   earlier "99% length-1" claim was wrong — it confused the dead-end
   *termination* rate (≈100% for any forward time-respecting walk, since every
   walk eventually dead-ends) with walk *length*. Measured walk-length medians:
   wikipedia 6, reddit 6, stackoverflow 5, **mooc 13**; all datasets yield
   multi-step walks with ample material for a $w{=}5$–$10$ skip-gram. The
   embedding-quality red flag is dissolved.
5. **T4's benefit scales with time-density / walk length / edge reuse**, i.e.
   it is largest exactly where temporal embedding is most interesting. On the
   time-dense mooc (median walk 13, edge reuse up to 2702×) empirical T4 nearly
   *halves* the cross fraction (47%→27%); on short-walk, sparse-reuse
   stackoverflow it helps less (50%→40%).

**REVISED go/no-go (GREEN, with scope):**
- **Proceed.** Walks are non-trivial on every dataset, and empirical-T4 lowers
  the cross fraction on all of them — most on the time-dense graphs that are
  the natural target for temporal embedding (mooc 47%→27%).
- **Position the thesis on time-dense temporal graphs** (long walks, edge
  reuse), where co-shard + empirical-T4 is strongest. Treat large
  sparse-temporal graphs (stackoverflow, residual ~40%) as the honest hard
  case, not the headline.
- **T4 is a hard prerequisite and a real build** (pilot sampling → edge weights
  → re-partition) — promote it to a co-equal contribution and budget for it;
  the trivial analytic weight does **not** work.
- **Scale-up validation still wanted:** confirm on a larger time-dense graph
  (e.g., TGB tgbl-comment) at higher rank counts. mooc is the in-hand
  time-dense proof but is small and bipartite.

**Artifacts (this machine):** `pilot_edge_weights.py` (pilot, now also reports
walk-length distribution), `partition_metis.py` (accepts `<scale>` arg or
`EWEIGHT_FILE` env), `e2_negsample.py` (E2), `.venv_part/` (pymetis venv,
git-ignore), weighted partitions under `data/<P>/<base>_tw.*`, weights at
`data/<base>.txt.ew`.

**E2 — results (2026-06-14, single machine, dim=64, 4 shards (balanced
random), window 5, K=5 negatives, chronological 80/20 split).** Temporal
link-prediction AP under four negative-sampling regimes (same walks, pairs,
init, RNG — only the negatives differ):

| dataset | global (ref) | local (no corr.) | local+weight | local+weight+exchange ρ=0.1 |
| --- | --- | --- | --- | --- |
| wikipedia | 0.856 | 0.709 | 0.813 | **0.840** |
| reddit | 0.931 | 0.701 | 0.779 | **0.889** |
| mooc | 0.600 | 0.550 | 0.579 | 0.531 |

**Findings:**
1. **The bias is real:** naive shard-local negatives (no correction) lose a lot
   — wikipedia AP 0.86→0.71, reddit 0.93→0.70. T3 is solving a genuine problem.
2. **The correction works** where link prediction has real signal: on wikipedia
   and reddit, local+weight+exchange recovers to ≈global (0.840/0.856 and
   0.889/0.931; AUC matches or exceeds global).
3. **The periodic cross-shard exchange is the load-bearing piece, not the weight
   alone.** On reddit the importance weight alone reaches only 0.779; adding the
   ρ=0.1 exchange jumps it to 0.889. This **confirms the plan's claim that the
   exchange-frequency bias–variance analysis is the core T3 contribution.**
4. **mooc is inconclusive — eval setup, not method:** global itself is near
   random (AUC 0.56), so the simple dot-product + random-negative LP protocol
   does not suit mooc's bipartite structure. Needs a user/item-aware eval before
   any mooc conclusion.

**PLAN IMPACT:**
- **T3 is GREEN** on graphs with LP signal, with the caveat that **the exchange
  (not just the weight) is essential** — the contribution centers on the
  exchange-rate ρ trade-off.

**⚠ CORRECTION (2026-06-14, after fixing the LP eval).** The AP numbers above
came from a **saturated** eval (random all-node negatives + global AUC/AP). A
proper **destination-ranking eval** (type-aware negatives for bipartite graphs;
MRR / Hits@10) is far more discriminative and **overturns the "≈global at
ρ=0.1" conclusion**:

| reddit, MRR | global | ρ=0 | ρ=0.1 | **ρ=0.3** | ρ=1.0 |
| --- | --- | --- | --- | --- | --- |
| MRR | 0.71 | 0.31 | 0.37 | **0.69** | 0.71 |

- ρ=0.1 recovers only ~half the MRR gap; **ρ≈0.3 is the real quality-preserving
  point** (MRR 0.69 ≈ global 0.71). Coupled with F2, ρ=0.3 gives **~3.2× comm
  reduction at matched quality** (not 8.4× at ρ=0.1, which loses quality).
- The eval fix also makes mooc interpretable (still weak: global MRR 0.065,
  |dst|=97 — confirming mooc is a poor embedding vehicle, not a method failure).
- **The destination-ranking eval is now the standard** for all quality numbers
  (`eval_lp` in `e2_negsample.py`, used by `inc2`/`m1_eval_lp`).

**⚠⚠ SECOND CORRECTION (2026-06-14, METIS shards instead of random).** E2's
shards were RANDOM; the engine uses METIS. Re-running with METIS shards
**reverses the ρ story entirely** (reddit, 3 seeds):

| reddit, METIS shards | global MRR | **local (ρ=0, no correction)** MRR |
| --- | --- | --- |
| seed 0 / 1 / 2 | 0.710 / 0.710 / 0.705 | **0.675 / 0.687 / 0.694** |

- **Pure shard-local negatives (ρ=0, no importance weight, no exchange) reach
  ~96–97% of global MRR** under METIS sharding — stable across seeds.
  On wikipedia, local even ≥ global (within noise).
- **Why:** graph-clustered (METIS) shards make a node's local negatives its
  graph neighbours = useful *hard negatives*. The bias that crippled local
  sampling under RANDOM shards (MRR 0.23) is an artifact of bad partitioning,
  not a real cost.
- **Consequence — T3 simplifies and the comm win grows:** under the engine's
  real partitioning, drop the importance-weight + ρ-exchange machinery entirely.
  ρ=0 ⇒ **no remote-negative traffic at all ⇒ the full ~17–41× embedding-comm
  reduction (F2 "bound" column) — now at near-global quality**, not the 3.2× of
  the random-shard ρ=0.3 detour.
- **Beautiful T3–T4 coupling (the paper's real story):** the same graph-aware
  (METIS/T4) partition that minimizes migration *also* makes shard-local
  negatives effective. One mechanism, two payoffs.
- Honest caveats: a small consistent gap remains on reddit (~3–5% MRR); a tiny
  ρ could close it but is marginal. Confirm on more datasets + with the
  in-engine embeddings (m1) before finalizing; mooc still degenerate.

> Reproduce: `.venv_part/bin/python e2_negsample.py data/<ds>.txt 4 64 3`

**E2 — the scientific claim of Innovation 2 / T3 (single-machine ML, no MPI).**
On one dataset, simulate sharding and train CTDNE-style embeddings three ways,
compare temporal link-prediction AP:
- (a) global negatives (reference),
- (b) shard-local negatives + importance weight + periodic exchange,
- (c) shard-local negatives, no correction (control).
- **Pass:** (b) matches (a) and clearly beats (c) ⇒ T3 works.
- **Fail:** (b) lags (a) ⇒ redesign the correction before any distributed build.
- Days of work, **no distributed system involved**.

### Gate
Proceed to the full build (M1+) **only if Phase 0 clears and E1, E2 both pass.**
This converts the riskiest, most expensive uncertainties into a 2–3 week,
mostly-reused-code check instead of a months-in mistake.

### Outline-driven from here
Unlike v1's exploratory data collection, v3's narrative is known in advance
(co-shard saves traffic + out-of-memory capability + TGB scores). Fix the three
§5 headline figures first, then run **exactly** the experiments that produce
them — more efficient than collecting broadly and writing later.

---

## 9. Milestones (gated on Phase 0–1 passing)

1. **M1** — co-shard + fused positive-pair path on multi-rank single node; vs v1
   two-stage → validate fusion throughput win.
2. **M2** — T3 shard-local negatives + correction → AP matches global negatives.
3. **M3** — multi-node + T4 temporal partition → migration & embedding traffic
   drop together.
4. **M4** — synthetic OOM-scale demo + TGB leaderboard numbers.
- **Target:** MLSys / KDD-applied / VLDB; fallback Cluster, SC ML track.

The weakest / most valuable piece is **T3 (the exchange-frequency bias–variance
analysis)** — it decides whether this reads as engineering or as a
systems×ML contribution with real analysis.

---

## 10. Execution plan (outline-locked, 2026-06-14)

Phase 0 is complete: novelty repositioned (temporal extension of NOMAD; do NOT
claim co-shard/Alltoallv/staleness as new — see `nomad_differentiation.md`), and
baselines secured (DistGER released; NOMAD as ablation config). This section
locks **claims → headline figure → experiment → falsifiable gate** so the build
produces only what each figure needs. Build the thinnest slice per figure,
measure, then decide to continue.

### Headline claims → figures → experiments

| # | Claim (what the paper asserts) | Headline figure | Experiment | Pass / fail gate |
| --- | --- | --- | --- | --- |
| **CA** | Temporal-reachability partitioning (T4) cuts cross-rank traffic vs static partitioning — *strongest evidence in hand* | F1: cross-rank pair fraction & comm volume, static-METIS vs T4, per dataset/window | T4 wired into the live engine; re-run E1 (`E1_WINDOW`) | T4 reproduces the offline drop **in the live engine** (e.g. stackoverflow $w{=}5$ ≥ −8 pts; mooc ≈ −20 pts) |
| **CB** | Co-shard + fused sampling–training cuts communication vs two-stage and vs NOMAD-style remote-context fetch | F2: comm volume & runtime — two-stage vs fused; vs NOMAD-style static config | M1 fused engine; instrument bytes/messages | fused < two-stage comm on ≥2 datasets, **embedding quality unchanged** (AP within noise) |
| **CC** | Temporal-aware embedding beats static SOTA on temporal link prediction | F3: temporal-LP AP — v3 vs DistGER(static) vs single-machine CTDNE; + neg-sampling ablation | run DistGER (released) static; v3 temporal; E2 scaled to the engine | v3 temporal-LP **> DistGER-static**, and shard-local+correction within ~2 AP of global-negatives (per E2) |
| **CD** | Engine scales on a real cluster; comm/compute crossover matches a cost model | F4: strong/weak scaling + comm fraction vs ranks | multi-node runs; fit α–β model | positive scaling in compute-heavy regime; model predicts crossover within tolerance |

Three secondary results already in hand (cite, don't re-run as headline): the
window-$w$ cross-fraction curve (E1), walk-length distributions, the ρ
exchange-rate trade-off (E2).

### Baselines (three-tier, locked)
1. **DistGER** (released, VLDB'23, static) — published SOTA, run treating the
   temporal graph as static.
2. **NOMAD-style static** — our engine with timestamps ignored + remote-negative
   fetch; ablates the temporal pieces.
3. **v1 two-stage** (sample-then-train) — ablates fusion.

### Milestones (refines §9; each gated on producing its figure)

| M | Build (thinnest slice) | Produces | Gate to continue |
| --- | --- | --- | --- |
| **M1** | co-shard embedding table + fused positive-pair training on multi-rank **single node** (reuse `walker`/`scheduler`/`comm_batch`) | F2-pilot | fused < two-stage comm, quality unchanged → else stop & rethink fusion |
| **M2** | T4 into the engine (pilot weights → partition load path) | F1 | live-engine T4 drop matches offline → else debug weighting |
| **M3** | temporal-LP harness + DistGER baseline + shard-local negatives in-engine | F3 | v3 ≥ DistGER-static on temporal-LP → else reconsider thesis |
| **M4** | real multi-node cluster + (synthetic OOM graph or TGB tgbl-comment) | F4 | positive scaling / model holds |

### Standing scope & rigor rules (from de-risking)
- **Primary datasets = time-dense** (mooc-like, TGB tgbl-comment); stackoverflow
  is the honest hard case. Always report walk-length distributions.
- **Fix the LP eval** (time-aware negatives; user–item scoring for bipartite)
  before trusting numbers — current dot-product+random-neg eval is degenerate on
  mooc.
- **Rigor before claiming:** multiple seeds + variance; METIS shards (not just
  random) for E2; ρ sweep.
- **Do NOT** present co-shard / batched-`Alltoallv` / bounded-staleness as
  contributions (NOMAD). Lead F1 (T4) — it is the strongest in-hand evidence.

### Definition of done for "starting experiments"
"Experiments" now means **building M1's thin slice and producing F2-pilot.**
Success criterion: a single bar chart showing fused co-shard comm < two-stage
comm on ≥2 datasets with matching AP. If that holds, proceed to M2; if not,
the fusion premise is wrong and we stop before sinking the full build.

### M1 result — F2-pilot (2026-06-14): GATE PASSED

Thinnest slice taken: instead of a full distributed-SGD engine, an
**embedding-communication accounting** on the real engine's real walks +
partition (extends the E1 window post-process in `main.c`; env `E1_DIM`,
`E1_NEG`, `E1_RHO`). Empirical inputs = cross-rank pair count + total pairs at
the training window; the rest is a transparent byte model (float32 vectors;
two-stage = NOMAD-style fetch+delta for cross context pairs and remote
negatives; fused = piggybacked positive grad + shard-local negatives with a
ρ-fraction still exchanged). Bytes, not wall-clock.

Embedding-comm at $w{=}5$, $d{=}128$, $K{=}5$, np=4, static partition, at the
**quality-preserving** exchange rate **ρ=0.3** (see the corrected ρ-frontier
below; the earlier ρ=0.1 was NOT quality-preserving):

| dataset | two-stage | fused @ρ=0.3 (quality-preserving) | fused @ρ=0 (bound, poor quality) |
| --- | --- | --- | --- |
| wikipedia | 160 MB | 49.4 MB (**3.2×**) | 3.9 MB (41×) |
| reddit | 198 MB | 61.6 MB (**3.2×**) | 5.6 MB (35×) |
| stackoverflow | 157 MB | 50.7 MB (**3.1×**) | 9.3 MB (17×) |
| mooc | 459 MB | 147.9 MB (**3.1×**) | 25.4 MB (18×) |

**Verdict: PASS (honest headline ~3.2×)** — fused < two-stage on all datasets.
At the quality-preserving operating point the embedding-comm reduction is
**~3.1–3.2×** (not the 6.8–8.4× first reported at ρ=0.1, which a proper eval
showed sacrifices quality). → proceed.

**Corrected ρ-frontier (reddit, destination-ranking MRR; see §8 E2 correction):**

| ρ | MRR | comm reduction |
| --- | --- | --- |
| 0.0 (pure local) | 0.31 | 35× |
| 0.1 | 0.37 | 8.2× |
| **0.3** | **0.69 ≈ global 0.71** | **3.2×** |
| 1.0 | 0.71 | ~1× |

**Honest decomposition:**
- The comm win is **dominated by shard-local negatives (T3)**, not the
  positive-pair piggyback (`remote_neg` ≈ 18× `cross_pairs`). The paper's
  communication story is **T3 + T4**, not the migration-piggyback.
- ρ is the headline knob and the **core T3 contribution**: it trades comm
  against quality. The honest sweet spot is **ρ≈0.3 → ~3.2× comm at ≈global
  quality**; ρ=0.1 over-saves comm at a real MRR cost; ρ=0 is poor.
- A reviewer will compare this to NOMAD's batch-all-global-negatives (no bias,
  but pays the full remote-negative comm = the two-stage column). Our claim:
  **~3× less embedding-comm at matched quality** via ρ-tuned shard-local
  negatives. That is the defensible contribution.

**⚠⚠ UPDATE — the ρ=0.3 figure above is superseded (see §8 second correction).**
The ρ-frontier was measured under RANDOM shards. Under the engine's real **METIS
shards**, pure shard-local negatives at **ρ=0** already reach ~96–97% of global
MRR (graph-clustered shards → local negatives are useful hard negatives). So the
honest operating point is **ρ=0**, giving the **full ~17–41× embedding-comm
reduction (F2 "bound" column) at near-global quality** — and the
importance-weight + exchange machinery can be dropped. The headline is therefore
**~17–41× (ρ=0, METIS shards)**, not 3.2× (which was a random-shard artifact),
with a small ~3–5% MRR caveat to confirm on more datasets / in-engine.
**T3 collapses into T4:** good partitioning is what makes local negatives work.

**Caveats:** byte model not wall-clock (M2+ engine needed for timing); excludes
walk-migration bytes (common to both); assumes balanced shards for the
remote-negative fraction $(P{-}1)/P$. New artifact: F2 accounting in `main.c`
(env-gated, off by default).

### M2 result — increment 1: real in-engine co-shard training (2026-06-14)

First real (not modeled) embedding training inside the MPI engine. New module
`embed.{c,h}`: per-rank co-located embedding table (indexed by local node id),
in-walk SGNS trained on **local-run window pairs** in the drive-to-death loop,
**shard-local negatives**. Env-gated (`EMBED_DIM/WIN/NEG/WNEG/LR`); off by
default. Increment-1 scope: cross-partition pairs **dropped** (added later via
piggyback), `wneg=1.0` (correction off). Shards dumped per rank; offline
temporal-LP eval in `m1_eval_lp.py`.

**Leakage-free temporal LP (wikipedia, train-only 80% partition, np=4, d=64,
win=5, K=5, 60k walkers):** **AUC 0.882, AP 0.895** on held-out future edges.

- Validates the full path end-to-end: co-shard table → in-walk SGNS →
  shard-local negatives → dump → eval. **GATE PASS.**
- Quality is strong even with cross-pairs dropped + no correction, because
  wikipedia's train-partition cross fraction is only ~5.6% — consistent with
  the co-shard thesis (low-cross ⇒ local-only training nearly suffices).
- Not directly comparable to E2's absolute numbers (different corpus size,
  single online pass vs 3 epochs, different eval negatives) — read it as "the
  in-engine embeddings are strong," not as a head-to-head with E2.

**Remaining increments (real engine):**
- **inc-2:** cross-partition pairs via migration-piggyback (walker carries
  recent context vectors); measure quality recovery on higher-cross graphs.
- **inc-3:** importance weight + periodic exchange (ρ) in-engine; reproduce E2.
- **inc-4:** wire training into the batched loop; measure **real** comm
  (wall-clock F2, not the byte model) two-stage vs fused.
- **F3/F4:** DistGER head-to-head (released code) on temporal-as-static;
  multi-node scaling.

**Artifacts:** `embed.{c,h}`, `m1_eval_lp.py`, `data/<ds>_train.txt`,
`log/embed_<ds>_p<P>_r*.txt`.

### M2 inc-4 — co-shard training in the batched loop (2026-06-14)

Moved fused training from drive-to-death into the **batched scheduler**
(`process_bucket`) — the real engine design. Per-walker local-run ring now
lives in `walker_t` (reset on migration; cross-partition pairs dropped, per the
inc-2 decision). Given the METIS-shard finding, the design is the **simplest
form**: co-shard table + in-walk SGNS + **shard-local negatives, no correction,
no exchange** (ρ=0).

**In-engine temporal LP (reddit, batched single-bucket, train-only 80%
partition, d=64, win=5, K=5, 60k walkers, ONE online pass):** AUC 0.853,
MRR 0.534, Hits@10 0.746.

- **Corroborates the simplified design in the real engine**: METIS partition +
  local-only negatives produces useful embeddings (Hits@10 0.75 over |dst|≈981).
- Absolute MRR is below the 3-epoch single-machine sim (~0.69) because the
  engine does a **single online pass** and **drops cross-partition positive
  pairs** (inc-1) — a training-budget gap, not a negative-sampling problem.
  Multi-epoch (re-run the walk job into the same table) would close it.

**Real wall-clock comm (two-stage vs fused) is DEFERRED, by design:** with
local-only negatives the fused engine has ~zero embedding communication (only
the sampler's existing walker migration), so a wall-clock comparison needs (a) a
NOMAD-style remote-exchange baseline and (b) a real multi-node network —
single-node intra-node MPI mutes it (v1's known caveat). The network-relevant
comm claim is already carried by the F2 byte model + the ρ=0/METIS quality
result (**~17–41× embedding-comm at near-global quality**). New artifact:
`walker_t.emb_ring`; training now active in both run loops.

**Engine status:** the fused design is essentially complete (co-shard + batched
+ local negatives). Remaining for a full paper: multi-epoch option; DistGER
head-to-head (released code, static baseline); multi-node scaling (F4).

### F3 baseline — DistGER unbuildable here; native static-walk baseline instead (2026-06-14)

**DistGER does not build on this machine:** it hard-requires Intel **MKL**
(`-D USE_MKL`, links `mkl_intel_ilp64`/`iomp5`, hardcoded
`/opt/intel/oneapi/mkl/2022.0.2`) + MPICH/Ubuntu. MKL is **x86-only**; we are on
**arm64 (Apple Silicon)** — no MKL exists. So the released static baseline is
infeasible locally (would need an x86 Linux cluster).

**Pivot (cleaner anyway):** added a `STATIC_WALK` mode to our own engine that
ignores the $t>t_{\mathrm{cur}}$ constraint (samples all neighbours = DeepWalk-
style static walks) on the *same* engine/partition/training. This isolates the
value of the temporal constraint apples-to-apples.

**Result (reddit, train-only, temporal LP):**

| walks | AUC | MRR | Hits@10 |
| --- | --- | --- | --- |
| temporal (15k walkers) | 0.854 | 0.534 | 0.750 |
| static (15k walkers) | 0.878 | 0.603 | 0.804 |
| **temporal (60k, matched pair budget)** | 0.877 | **0.629** | 0.793 |
| static (15k, ref) | 0.878 | 0.603 | 0.804 |

- Naively static *beats* temporal — but that is a **training-volume confound**:
  static walks never dead-end (full 30 steps) so they yield far more pairs.
- **At matched training budget, temporal ≈ static** (MRR 0.629 vs 0.603,
  Hits@10 0.793 vs 0.804 — within noise).

**⚠ IMPORTANT framing consequence:** the time-respecting constraint does **not**
improve temporal-LP quality over static walks (it is on par at equal budget).
**Do NOT claim "temporal embeddings are better."** v3's defensible value is
**systems/enabling**: provide the time-respecting walk *primitive* (required by
CTDNE/CAW-style temporal methods) at distributed scale, **at no quality cost vs
static**, plus the T3→T4 communication finding. Quality superiority is not a
contribution; the temporal primitive + the comm story is.

New artifact: `STATIC_WALK` env in `walker.c`.

### Time-sensitive eval (the last shot at a quality claim) — NEGATIVE (2026-06-14)

Added a **time-local-negative** eval (`m1_eval_lp.py`): for each test edge
(u,v,t), negatives = destinations active in the SAME time window as t (20 bins)
— tests fine-grained temporal discrimination that global-negative LP hides.
Compared temporal vs static walks at matched training budget (reddit):

| walks | GLOBAL MRR | TIME-LOCAL MRR | TIME-LOCAL Hits@10 |
| --- | --- | --- | --- |
| temporal (60k) | 0.619 | 0.574 | 0.773 |
| static (15k) | 0.606 | 0.562 | 0.789 |

**Still a wash** — temporal ≈ static even on the time-sensitive eval (temporal
MRR marginally higher, Hits@10 marginally lower; within noise). **The quality
probe FAILED to find a temporal advantage.**

Likely root cause: both produce a **single static vector per node**, which
cannot encode time-varying behavior — so the walk-sampling difference washes out
at inference. A genuine temporal-quality benefit would need **time-aware
embeddings** (TGN-style time encoding), i.e., a different model, not walk+skip-
gram. Caveats: one dataset with LP signal (reddit); our walks are forward-uniform
+ dead-ending (not CTDNE/CAW biased walks); coarse bins.

**DECISIVE FRAMING (post-de-risking).** v3 is a **systems / enabling**
contribution, **not** a quality-improvement one:
- Provides the distributed time-respecting walk **primitive** (required by
  temporal methods) at scale, **at no quality cost vs static** walks.
- Plus the **T3→T4 communication finding** (good partition makes shard-local
  negatives effective; ~17–41× embedding-comm at near-global quality vs a
  two-stage baseline).
- Systems mechanics overlap NOMAD (static, 2026); quality is on par with static.

Realistic venue: **mid-tier (Cluster / ICPP / IPDPS)** as an enabling-systems
paper. Not a strong-venue quality story. Decision for the author: (B) write the
modest enabling-systems paper, or (C) pivot (v2 pure-HPC, or a time-aware-model
direction that could claim quality). De-risking has made this call cheap and
clear before any months-long build.

### Historical-negative eval (the field-standard hard test) — temporal edges out static (2026-06-14)

Added historical-negative LP (Poursafaei et al., NeurIPS'22 D&B): negatives =
u's TRAIN-period partners other than the true v. This is the eval where easy
negatives stop hiding differences (the EdgeBank lesson). Reddit, matched budget:

| eval | temporal (60k) MRR | static (15k) MRR | Δ |
| --- | --- | --- | --- |
| global-neg | 0.618 | 0.604 | +0.014 |
| time-local-neg | 0.575 | 0.564 | +0.011 |
| **historical-neg** | **0.135** | **0.127** | +0.008 |
| historical Hits@10 | 0.170 | 0.159 | +0.011 |

**Findings:**
1. Historical negatives are far harder (MRR ~0.6 → ~0.13) and both static-vector
   approaches are weak there — consistent with the field: ranking the *current*
   partner above *past* partners needs **time-aware models** (TGN), which a
   single static vector per node cannot be. This is the ceiling, not a bug.
2. **Temporal walks give a small but CONSISTENT edge over static** across all
   three eval regimes (+0.008–0.014 MRR, ~5–7% relative; temporal ≥ static in
   5/6 metrics). Not a blowout, but a real, literature-aligned positive.

**Refined B framing (slightly upgraded):** not merely "no quality cost" but a
**small consistent quality edge** from the temporal constraint, *plus* the
enabling systems contribution, *plus* the honest scoping that big temporal gains
require time-aware models (out of scope for walk+skip-gram). This matches the
post-2022 literature (CTDNE's temporal>static is real but eval-dependent;
EdgeBank/TGB show easy negatives hide differences and simple baselines are
strong).

**Caveat:** Δ is small (0.008–0.014 MRR), single seed / single dataset
(reddit) — direction is consistent across 3 evals but **needs multiple seeds +
more datasets to claim significance**. New artifact: historical + time-local
evals in `m1_eval_lp.py`.

### Multi-seed × multi-dataset confirmation (2026-06-14) — the real story emerges

3 seeds (`RNG_SEED` env), temporal (60k) vs static (15k), matched budget:

| dataset | eval | temporal (μ, 3 seeds) | static (μ) | Δ |
| --- | --- | --- | --- | --- |
| reddit | global MRR | 0.622 | 0.604 | temporal +0.018 |
| reddit | **historical MRR** | 0.124 | 0.122 | ~tied |
| wikipedia | global MRR | 0.652 | **0.739** | **static +0.087** |
| wikipedia | **historical MRR** | **0.132** | 0.094 | **temporal +0.038** |

**The eval determines the winner — exactly the EdgeBank/Poursafaei lesson:**
- On **easy (global random) negatives**, the high-data static walks can win big
  (wikipedia static +0.087) — easy negatives reward "is this a plausible edge
  at all," which longer static walks nail. Easy eval is **unreliable**.
- On the **field-recommended HISTORICAL negatives** (current vs past partners),
  **temporal walks win** — clearly on wikipedia (**+0.038, and the 3 seeds are
  NON-OVERLAPPING**: temporal min 0.122 > static max 0.108), tied on reddit.

**Verdict — a defensible, literature-aligned quality claim for B:**
> The time-respecting constraint improves **hard-negative** temporal LP
> (clearly on wikipedia, neutral on reddit), while easy-negative evals are
> unreliable and can favor static. This reproduces the Poursafaei/EdgeBank
> finding on our engine and is the honest, correct way to state the temporal
> benefit.

So B is **no longer "enabling-only"**: there is a real (eval-gated, dataset-
dependent) temporal quality benefit on the correct eval. Still mixed across
datasets (wikipedia clear, reddit tied) → **report both honestly; add 1–2 more
datasets** to map how often the wikipedia-style clear win occurs. New artifact:
`RNG_SEED` env in `main.c`.

### Clarity check — mooc added (2026-06-14): benefit is dataset-dependent, not universal

mooc, 3 seeds: temporal global MRR ~0.050 / hist ~0.098; static global ~0.060 /
hist ~0.165. **mooc is DEGENERATE and excluded** (objective grounds, not
cherry-picking): global MRR ≈ 0.05 is the **random floor** for |dst|≈97 → the
embeddings learn essentially nothing on mooc, so any eval on top is noise; and
with ~97 destinations + users repeating the same few actions, historical
negatives overlap the positive (ill-posed eval). Static's apparent historical
"win" on mooc is a degeneracy artifact.

**Honest tally across all usable temporal datasets:**

| dataset | hard (historical) eval | status |
| --- | --- | --- |
| wikipedia | **temporal > static** (+0.038, non-overlapping seeds) | clean ✓ |
| reddit | tied | clean |
| mooc | static > temporal | **degenerate** (random-floor embeddings) — exclude |
| stackoverflow | — | degenerate (undersampled, sparse history) |

**Measured verdict (pre-TGB):** the temporal hard-eval benefit is real but, on
local data, clear only on wikipedia (reddit tied; mooc/SO degenerate).

### TGB clarity check — tgbl-review REPRODUCES the pattern robustly (2026-06-14)

Downloaded TGB (`py-tgb` + torch) and exported **tgbl-review** (Amazon reviews,
4.87M edges, 350K nodes) to our format. At feasible walker budgets the large
graph is undercovered (global MRR ≈ random); with heavy coverage (temporal 2M /
static 500k walkers) it becomes functional and shows, across **3 seeds**:

| tgbl-review | global MRR | historical MRR |
| --- | --- | --- |
| temporal | 0.073–0.077 | **0.156–0.160** |
| static | 0.223–0.226 | 0.125–0.128 |

- **historical: temporal +0.031, NON-OVERLAPPING across 3 seeds** (temporal min
  0.156 > static max 0.128) — same as wikipedia, on a 25× larger independent
  TGB benchmark.
- global: static wins big (longer walks → more easy-task training) — the easy
  eval remains unreliable.

**UPGRADED cross-dataset verdict (the quality claim is now solid):**

| dataset | hard (historical) eval | robustness |
| --- | --- | --- |
| wikipedia | temporal **+0.038** | 3 seeds non-overlapping ✓ |
| tgbl-review (TGB, 4.87M) | temporal **+0.031** | 3 seeds non-overlapping ✓ |
| reddit | tied | clean |
| mooc / stackoverflow | degenerate | excluded (small-dst / undercovered) |

> **Temporal walks robustly beat static on the field-recommended hard-negative
> (historical) eval — on 2 of 3 clean datasets including a large TGB benchmark —
> while easy (global) negatives are unreliable and favor static.** This
> operationalizes the Poursafaei/EdgeBank lesson on our engine and is a genuine,
> eval-gated, cross-dataset temporal quality benefit. Honest scope: the benefit
> is specific to hard negatives; absolute historical MRR is low (~0.16) because
> single static-vector embeddings cap it — large absolute gains need time-aware
> models. Datasets with tiny destination sets (mooc) or that need cluster-scale
> coverage (stackoverflow) are out of reach here.

**This resolves the quality question for B**: not enabling-only, not a universal
quality win, but a **robust eval-gated temporal benefit on the correct eval,
reproduced across two independent datasets (one large, from TGB)**. New
artifacts: `py-tgb`+`torch` in `.venv_part`, `data/tgbl_review*.txt`,
`baselines/tgb_data/`.

### M2 inc-2 finding — value of training cross-partition pairs (2026-06-14)

Before building the expensive distributed migration-piggyback (carry context
vectors forward + route gradient deltas back — the hardest engine component),
quantified its value single-machine under the **real METIS partition** via
`inc2_crosspair.py`: train SGNS with global negatives two ways on the same
walks/partition — **keep** all window pairs (= inc-2) vs **drop** cross-shard
pairs (= inc-1 local-only) — and compare temporal-LP AP.

| dataset | cross-pair % | keep AP | drop AP | gap |
| --- | --- | --- | --- | --- |
| wikipedia | 16.9% | 0.856 | 0.842 | **−1.4 pt** |
| reddit | 23.4% | 0.931 | 0.906 | **−2.5 pt** |
| stackoverflow | 41.2% | 0.592 | 0.589 | −0.3 pt (degenerate: AUC≈0.55, near random) |

**Verdict: DEFER / likely SKIP inc-2.** On the graphs where LP has signal,
dropping cross-partition positive pairs costs only 1.4–2.5 AP — local-only
(inc-1) captures nearly all the quality (cross pairs are the minority and
skip-gram is robust to dropping some context). The migration-piggyback is the
most complex piece of the engine; this gap does not justify it yet.

**Residual uncertainty (honest):** no clean *high-cross + LP-has-signal*
measurement exists — stackoverflow and mooc LP are both degenerate under the
current dot-product + random-negative eval. So "skip inc-2" is supported but not
airtight. **Gate before fully committing to skip:** fix the LP eval (time-aware
negatives; user–item scoring for bipartite) and re-measure keep-vs-drop on a
time-dense, high-cross graph. If the gap stays small there too, drop inc-2 for
good and ship local-only co-shard (a major engine simplification).

**Revised increment ladder:** inc-2 deprioritized → next highest value is
**(a) fix the LP eval** (unblocks mooc/time-dense quality numbers and the inc-2
gate), then **(b) inc-4** (wire training into the batched loop; measure *real*
wall-clock comm two-stage vs fused), then **F3/F4** (DistGER head-to-head;
multi-node scaling). New artifact: `inc2_crosspair.py` (walk-capped for large
graphs).
