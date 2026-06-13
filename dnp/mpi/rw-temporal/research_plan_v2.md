# Research Plan v2 — Moving-Time-Window Locality for Distributed Temporal Walk Sampling

Target: a top-tier HPC-systems venue (SC / IPDPS; fallback Cluster / ICPP).
Builds on the accepted NBiS 2026 paper (the "v1" characterization study).

> **中文导航(这份文件是什么)**
> NBiS 那篇是"测量研究":证明**通信粒度**(而非调度)主导分布式时序游走成本,
> 批处理迁移比逐 walker 发送快 78–347×。这份 v2 是**下一篇的研究计划**,
> 经 `/deep-research`(2022–2026 前沿核查)修正后重写:
> - **空白成立**:没有系统做"分布式连续时间时序游走"。
> - **但通用机制都被占了** —— 加权分区(WASP'21、DistGER MPGP VLDB'23)、
>   在线重分区(Balanced RePartitioning DISC'16+)、层次聚合(MT-lib'21、Open MPI HAN)、
>   α-β 成本模型,都是已有工作。
> - **唯一防得住的新颖性轴 = 时序特化**。本计划把全部赌注押在一条主线:
>   **"游走的活跃前沿是一个随 $t_{\mathrm{cur}}$ 推进的移动时间窗"** —— 没有现有工作建模这一点。

---

## 0. What changed from v1 (the original four-pillar plan)

The first draft pitched four equal contributions: (1) async + distributed
termination detection, (2) temporal-aware partitioning, (3) topology-aware
hierarchical aggregation, (4) an α-β cost model. The deep-research pass
(2022–2026 frontier) found three of the four are **generic mechanisms that
already exist**, and one rests on a **correctness argument that monotone time
makes moot**. The rewrite below:

| v1 contribution | verdict | v2 disposition |
| --- | --- | --- |
| (2) traversal-weighted partitioning | generic idea exists (WASP'21, DistGER MPGP'23); online repart. exists (BRP DISC'16+) | **absorbed** into the single thesis, scoped to its *temporal* form |
| (3) hierarchical intra→inter-node aggregation | established (MT-lib'21, Open MPI HAN'20, node-aware Allreduce SC19) | **demoted** to an implementation/optimization subsection, not a claim |
| (4) α-β cost model | standard model, routine instantiation | **demoted** to analysis/validation, not a claim |
| (1) async + termination detection | KnightKing/DistGER lack it, but the techniques are textbook; **monotone time bounds walk length, so the "bounce-back livelock" risk is largely moot** | **repositioned**: sell comm/compute *overlap + load balancing under temporal skew*, not termination-detection correctness |

---

## 1. The single thesis (the only defensible novelty axis)

**No prior system models the active frontier of a temporal walk as a
*moving time window* that advances with the cursor $t_{\mathrm{cur}}$.**

In a static distributed walk (KnightKing, DistGER, ScaleRunner) every edge of
every partition can be touched at any time, so partitioning and scheduling are
*time-invariant*. In a **time-respecting** walk the cursor only moves forward
(NBiS §5.2: each step ~halves the remaining interval; 100% dead-end by ~20
steps). Therefore at any wall-clock moment the *hot* set of edges — the ones a
walker could actually traverse next — is a **narrow, monotonically advancing
slice of the timeline**, not the whole graph.

This one fact, never exploited before, lets us co-design three layers so the
active frontier stays rank-local:

- **Partition** for the *temporal* traffic a time-respecting walk generates,
  not the static cut.
- **Schedule** walkers by their cursor band so a rank works one time slice at a
  time (cache- and frontier-coherent).
- **Communicate** only the migrations that cross the *current* frontier,
  overlapped with the next slice's compute.

> **Reader take-away:** in temporal distributed walks, locality is a property
> of *time*, not of graph topology — and that is what static walk systems and
> static temporal-graph partitioners (WASP, SPEED, DistGER) all miss.

---

## 2. Positioning — what each related system does NOT do

(Verified in the deep-research pass; cite these explicitly in Related Work.)

| System | venue | scope | why it does not cover us |
| --- | --- | --- | --- |
| KnightKing | SOSP'19 | distributed RW, **static** | BSP, two sync rounds/step; no timestamps |
| DistGER (+MPGP) | VLDB'23 | distributed RW + RW-aware partition, **static** | proximity-weighted, computed upfront; no temporal/time-window |
| ScaleRunner | Euro-Par'25 | MPI RW engine, **static** | 1st/2nd-order DeepWalk/node2vec; no temporal semantics |
| TEA | EuroSys'23 | temporal RW, **single machine** | not partitioned across machines |
| TGL / DistTGL | VLDB'22 / SC'23 | temporal **GNN training** | memory module, not walk sampling |
| SPEED | 2023 (arXiv 2308.14129) | streaming **temporal** partition | optimizes replication/memory for training, not walk-migration traffic; no time-window frontier |
| WASP | 2021 | traversal-frequency-weighted partition, **static** | no temporal weighting, no frontier drift |
| Balanced RePartitioning | DISC'16 + follow-ons | online repartition **theory** | abstract collocation; not temporal/RW |
| MT-lib / Open MPI HAN | 2021 / 2020 | hierarchical intra→inter-node aggregation | generic collective mechanism (we *use* it, don't claim it) |

**The empty cell remains exactly ours:** distributed, continuous-time,
time-respecting walk sampling with a **time-window-aware** runtime.

---

## 3. Contributions (rewritten — three, all temporal-specific)

**C1. Moving-time-window execution model + temporally-skewed load balancing.**
A barrier-relaxed engine that advances walkers in cursor bands. Because time is
monotone, a rank's work for band $[\tau, \tau+\Delta)$ is finite and known, so
we overlap band-$k$ migration with band-$(k{+}1)$ compute (non-blocking /
RMA). The hard, *temporal* problem is **load skew**: as the frontier advances,
active vertices shift, so rank utilization is time-varying — we balance walkers
across ranks *as the window moves*. (We explicitly do **not** claim async
termination detection as novel; monotone time already bounds termination.)

**C2. Temporal-reachability-weighted partitioning.**
Weight each edge by the probability a time-respecting walk actually traverses
it — estimable cheaply from temporal degree + timestamp density, or one pilot
sampling pass. Partition to minimize **expected temporal migration traffic**,
not the static cut. Distinct from WASP/MPGP (static, proximity/visit-frequency
with no time axis) and SPEED (replication objective). Include a light
**frontier-drift rebalance** that re-homes only the vertices entering the hot
window.

**C3. A scaling characterization + validated cost model for the temporal
crossover.** Extend NBiS's communication-bound→compute-bound crossover to a
real multi-node cluster, and give a predictive model
$T \approx \max(\text{compute}, \alpha\,m + \beta\,b)$ specialized with the
*temporal* terms (band count, per-band crossing rate, dead-end-bounded steps).
Presented as **analysis/validation**, not as a novel model.

> Hierarchical intra→inter-node migration aggregation (old C3) appears as an
> **implementation optimization** under C1 — credited to MT-lib / HAN, not
> claimed.

---

## 4. Technical thrusts (actionable, tied to the existing codebase)

### T1 — Moving-window scheduler + overlap (implements C1)
- Replace the single global round with **cursor-band rounds**: process all
  walkers with $t_{\mathrm{cur}} \in [\tau, \tau+\Delta)$, then advance $\tau$.
  (Reuses the existing time-window scheduler `scheduler.{c,h}`, now as the
  *primary* execution model, not an ablation.)
- Overlap: post band-$k$ outbound migrations with `MPI_Isend`/`MPI_Ialltoallv`
  (or RMA `MPI_Put` into remote band buffers) and step band-$(k{+}1)$ locally
  while they fly.
- Load balancing: per-band walker counts per rank are measured; over-loaded
  ranks shed walkers to idle ranks (work-stealing keyed by cursor band).
- Aggregation optimization: `MPI_Comm_split_type(SHARED)` to coalesce
  intra-node migrations before one inter-node message per destination node
  (cite MT-lib/HAN).

### T2 — Temporal-reachability partitioner (implements C2)
- Edge weight $w(u\!\to\!v,t) \propto$ Pr[a walk reaches $u$ before $t$] ×
  Pr[picks this edge]. Approximate from temporal in-degree-before-$t$ and the
  TAL suffix size. Feed weights to METIS (`partition_metis.py`).
- Frontier-drift rebalance: recompute hot-window ownership every $K$ bands;
  migrate only boundary vertices entering the window. Report rebalance cost.

### T3 — Cost model + scaling study (implements C3)
- Instrument α (per-message), β (per-byte) on the target cluster; plug in
  measured band count and per-band crossing rate; predict the np at which
  scaling turns positive; validate against strong/weak scaling.

---

## 5. Evaluation plan

| item | plan |
| --- | --- |
| **Cluster** | real multi-node (your center's machine), 4 → 8 → 16 → 32 → 64 *physical* nodes |
| **Datasets** | TGB (tgbl-comment, tgbl-flight), full Stack Overflow, GDELT; **+ a controlled synthetic billion-edge temporal graph** (TGB may be too small to show the crossover at scale — flagged by deep-research) |
| **Baselines** | TEA (single-node temporal, the must-have); **ScaleRunner (Euro-Par'25, newest MPI RW)**; KnightKing (static distributed); NBiS bulk-sync engine (ablation for the overlap win) |
| **Metrics** | strong/weak scaling, walks/sec, comm-time fraction, message count, **static cut vs temporal-weighted cut**, rebalance overhead, cost-model prediction error |
| **Ablations** | T1 (banded+overlap) / T2 (temporal partition) / aggregation on–off, isolated |
| **Headline figure** | NBiS's negative light-load scaling → turned positive after banded overlap; comm fraction ↓ with nodes; model prediction vs measured crossover |

---

## 6. Risks & mitigations

| risk | mitigation |
| --- | --- |
| **C1 novelty thin** if a temporal/dynamic RW engine already does async overlap | **must verify** FlashMob, ThunderRW, GraphWalker, and any 2023–26 TEA derivative before committing (deep-research did not clear these) |
| Monotone time makes termination trivial → C1 looks like "just scheduling" | lead C1 with **load-skew-under-drift**, the genuinely hard temporal problem; overlap is the mechanism, not the claim |
| Temporal-weight estimation cost eats the gains | cheap analytic/pilot estimate; **report estimation overhead separately** |
| Dead-end → short walks → throughput dominated by spawn/teardown | report **per-step** throughput; optionally recency-biased sampling to lengthen effective walks |
| TGB too small to exhibit crossover | synthetic billion-edge generator (above) |
| Reviewer: "C2 = WASP/MPGP with timestamps" | pre-empt: temporal-reachability weight + frontier-drift rebalance are *not* expressible in static visit-frequency; show a static-weight baseline loses |

---

## 7. Milestones & venue windows

1. **M1–2** — T1 banded+overlap engine; multi-node bring-up; NBiS-baseline win (turn negative scaling positive). *Sufficient alone for Cluster/ICPP.*
2. **M2–3** — T2 temporal partitioner + frontier rebalance; TEA + ScaleRunner baselines.
3. **M3–4** — cost model, billion-edge synthetic, full ablations.
4. **Anchor:** **SC 2027** (abstracts ~April). Fallback **IPDPS 2028 / Cluster 2027 / ICPP 2027**.

T1 + multi-node + model = a solid mid-tier paper; T1+T2+C3 together = SC/IPDPS.

---

## 8. Prior-art reading list (from the deep-research pass)

- KnightKing, SOSP'19 — distributed static RW (BSP, 2 rounds/step).
- DistGER + MPGP, VLDB'23 (arXiv 2303.15702) — RW-aware static partition, −45% comm.
- ScaleRunner, Euro-Par'25 — newest MPI RW engine; **add as baseline**.
- TEA, EuroSys'23 — single-machine temporal RW; **must compare**.
- SPEED, 2023 (arXiv 2308.14129) — streaming temporal partition (replication objective).
- WASP, 2021 (Data Sci. & Eng., Springer) — traversal-frequency-weighted partition.
- Balanced RePartitioning, DISC'16 + SPAA'23/STACS'24 — online repartition theory.
- MT-lib, 2021 (arXiv 2103.15024) — topology-aware intra→inter-node aggregation.
- Open MPI HAN (PR #7735, 2020) — hierarchical `COMM_TYPE_SHARED` collectives.
- TGB — Temporal Graph Benchmark datasets (github.com/shenyangHuang/TGB).

> Verification note: the load-bearing prior-art hits (WASP, DistGER/MPGP,
> MT-lib) are 3-0 confirmed; several hierarchical-aggregation items are
> "surfaced but unvoted" (session-limit abstains) — directionally reliable but
> re-check before citing as the decisive precedent.
