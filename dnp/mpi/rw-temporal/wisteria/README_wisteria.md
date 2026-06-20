# Wisteria/BDEC-01 multi-node run guide (F4 scaling + OOM capability)

Target: **Odyssey** subsystem (Fujitsu A64FX, aarch64, 48 cores/node, ~28 GiB
usable/node, Tofu-D interconnect). Scheduler: Fujitsu `pjsub`. This is the
remaining experiment for research_plan_v3.md (real multi-node wall-clock comm +
strong/weak scaling — the single-node intra-MPI byte model can't show it).

## 0. Key constraints (from the user guide)
- **Compute nodes CANNOT read `/home`.** Stage everything under
  `/work/<group>/<user>/` (FEFS, shared across all nodes, 2 TB). Submit from /work.
- aarch64 binary: build with `mpifccpx` (cross-compiler on the login node).
- `--mpi proc=<total>` is **mandatory** for multi-node (else 1 rank/node).
- Memory **~28 GiB/node** — size partition count so per-rank footprint fits.
- Launch via Fujitsu `mpiexec` inside a `pjsub` job (no plain mpirun).

## 1. Stage + build
```
# on a login node
cd /work/<group>/<user> && git clone <repo> && cd .../rw-temporal
sh wisteria/build_wisteria.sh          # -> ./rw (aarch64)
```

## 2. Partition each dataset into <proc> parts (proc = nodes*48)
The engine reads `data/<P>/<base>.sub<r>.txt` + `.rt<r>.txt`. Generate for every
proc count in the scaling sweep (96,192,384,768,1536):
```
# needs pymetis: python3 -m venv venv && venv/bin/pip install numpy pandas pymetis
for P in 96 192 384 768 1536; do
  venv/bin/python partition_metis.py data/<dataset>.txt $P
done
```
(Run in the `prepost` rscgrp or on a login node; large P × large graph is the
slow step. Partitions are reused across all runs, so do this once.)

## 3. Scaling sweep — submit one job per point
`wisteria/job_scaling.sh` (EDIT `-g GROUP` first; it sets `NO_LOG=1` so the
unscalable gather is skipped — only `PHASE`/`elapsed` are produced). Pass run
parameters as env vars; override node/proc on the pjsub command line.

**Weak scaling** (fixed walkers/rank, default `NWALK=50000`):
```
for n in 2 4 8 16 32; do
  pjsub -L node=$n --mpi proc=$((n*48)) wisteria/job_scaling.sh
done
```
**Strong scaling** (fixed TOTAL walkers; per-rank shrinks as P grows):
```
for n in 2 4 8 16 32; do
  TOTAL=2000000 pjsub -x TOTAL -L node=$n --mpi proc=$((n*48)) wisteria/job_scaling.sh
done
```
**The full F4 matrix** — set env per sweep (`-x VAR` forwards it to the job):
| sweep | env |
| --- | --- |
| weak, static partition | `DATASET=stackoverflow_a2q` |
| weak, **T4** partition | `DATASET=stackoverflow_a2q_tw` |
| strong, static | `DATASET=stackoverflow_a2q TOTAL=2000000` |
| strong, T4 | `DATASET=stackoverflow_a2q_tw TOTAL=2000000` |
| embedding comm: fused | `EMB=1 EMBMODE=fused` |
| embedding comm: two-stage | `EMB=1 EMBMODE=twostage` |

e.g. `DATASET=stackoverflow_a2q_tw pjsub -x DATASET -L node=8 --mpi proc=384 wisteria/job_scaling.sh`.
(Forward env to the batch job with pjsub `-x VAR1,VAR2` or `export` inside a wrapper.)

Smoke-test first in `debug-o` (≤144 nodes, 30 min) or interactively:
`pjsub --interact -g GROUP -L rscgrp=interactive-o,node=2 --mpi proc=96`.
Note: strong scaling at high P gives a small per-rank load (TOTAL=2M / 1536 ≈
1302 walkers/rank) → likely communication-bound at the high end; raise TOTAL
(e.g. 8–16M) if you want the compute-bound regime at scale.

## 4. F4 experiment protocol

**Metric source:** each run prints
`PHASE compute=.. exchange=.. allreduce=.. comm_frac=..%` and `elapsed=..`.

| study | what to fix | what to vary | reads |
| --- | --- | --- | --- |
| **Strong scaling** | total walkers (e.g. 2,000,000) → `NWALK=2000000/proc` | nodes 2→32 | elapsed, speedup vs 2 nodes, comm_frac |
| **Weak scaling** | walkers/rank (`NWALK=50000`) | nodes 2→32 | elapsed (ideal=flat), efficiency, comm_frac |
| **Comm fraction** | — | nodes 2→32 | `comm_frac` rises with nodes on real Tofu (the headline the single-node model couldn't measure) |
| **Walk vs walk+embed** | dataset, nodes | `EMB=0` vs `EMB=1` | elapsed delta = training cost; embedding comm ≈ 0 (local negatives) |

Datasets: `stackoverflow_a2q` (17.8M edges) and `tgbl_review` (4.87M) for
scaling. Set `DATASET=`, `NWALK=`, `EMB=` as env in the job or edit the script.

**OOM-capability demo (separate, the motivation):** a graph whose footprint
exceeds 28 GiB on few nodes but fits when partitioned across many. 17.8M-edge
stackoverflow fits one node, so it does NOT show this — generate a **synthetic
billion-edge temporal graph** (scale-free + timestamps) and show the engine
runs at high node count while low node count OOMs. (Generator TBD; note as the
capability figure.)

## 4b. Headline experiment: static vs T4 (temporal-reachability) partition

Weak scaling showed the system is communication-bound, driven by the high
cross-rank rate of static-METIS partitions. T4 weights edges by empirical
time-respecting traversal frequency to cut crossing → cut migration → cut comm.
Build a T4-partitioned copy and run the SAME sweep; compare `comm_frac`/elapsed.

On the **Mac** (`.venv_part` has pymetis):
```
# 1) pilot: per-edge traversal weights (once, on the full graph)
.venv_part/bin/python pilot_edge_weights.py data/stackoverflow_a2q.txt 500000 30
# 2) partition each proc count WITH those weights -> data/<P>/stackoverflow_a2q_tw.*
for P in 96 192 384 768 1536; do
  EWEIGHT_FILE=data/stackoverflow_a2q.txt.ew \
    .venv_part/bin/python partition_metis.py data/stackoverflow_a2q.txt $P
done
# 3) rsync the _tw partitions up
rsync -av data/96 data/192 data/384 data/768 data/1536 \
    z30130@wisteria.cc.u-tokyo.ac.jp:/work/gz00/z30130/<rw-temporal>/data/
```
On the **cluster**, run the sweep against the T4 dataset name:
```
for n in 2 4 8 16 32; do
  DATASET=stackoverflow_a2q_tw pjsub -L node=$n --mpi proc=$((n*48)) wisteria/job_scaling.sh
done
```
Compare `comm_frac`/`exchange`/`elapsed` of `stackoverflow_a2q` (static) vs
`stackoverflow_a2q_tw` (T4) at each node count → the T4 communication win on a
real network.

## 5. Honest scope (carry into the paper)
- Quality result is already established single-node (eval-gated temporal benefit,
  wikipedia + tgbl-review on historical negatives). Multi-node does not change it.
- Multi-node adds: real wall-clock comm + scaling + the OOM-capability demo.
- For the real **two-stage vs fused** comm comparison, the NOMAD-style remote-
  embedding-exchange baseline must be implemented (currently fused has ~zero
  embedding comm via local negatives; the baseline is not yet built). Without it,
  F4 shows fused scaling + walk-migration comm only — still a valid result.
