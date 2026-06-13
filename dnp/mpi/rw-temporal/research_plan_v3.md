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

## 7. Prior-art status (⚠ embedding line NOT deep-researched yet)

**Verified (from the v2 deep-research pass):**
- DistGER (VLDB'23, arXiv 2303.15702) — distributed RW→embedding, **static**; 2-stage.
- TGL (VLDB'22) / DistTGL (SC'23) — distributed temporal **GNN training**, not walk embedding.
- SPEED (2023, arXiv 2308.14129) — streaming temporal partition, replication objective.
- TEA (EuroSys'23) — single-machine temporal RW.

**Must verify before committing (the most direct novelty threats):**
1. Any **distributed** streaming/incremental temporal node-embedding system
   (dynamic DeepWalk, tNodeEmbed, online CTDNE variants) — single-machine or distributed?
2. Distributed embedding-consistency work (Hogwild!, parameter-server, async
   word2vec) **specialized to temporal-walk access patterns** — has anyone co-sharded
   graph + embedding for *temporal* walks?
3. Whether the migration-piggyback idea (positive-pair gradient on the migration
   message) appears in any distributed graph-embedding system.

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
