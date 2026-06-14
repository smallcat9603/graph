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
`wisteria/job_scaling.sh` (EDIT `-g GROUP` first). Override node/proc per point:
```
for n in 2 4 8 16 32; do
  pjsub -L node=$n --mpi proc=$((n*48)) wisteria/job_scaling.sh
done
```
Smoke-test first in `debug-o` (≤144 nodes, 30 min) or interactively:
`pjsub --interact -g GROUP -L rscgrp=interactive-o,node=2 --mpi proc=96`.

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

## 5. Honest scope (carry into the paper)
- Quality result is already established single-node (eval-gated temporal benefit,
  wikipedia + tgbl-review on historical negatives). Multi-node does not change it.
- Multi-node adds: real wall-clock comm + scaling + the OOM-capability demo.
- For the real **two-stage vs fused** comm comparison, the NOMAD-style remote-
  embedding-exchange baseline must be implemented (currently fused has ~zero
  embedding comm via local negatives; the baseline is not yet built). Without it,
  F4 shows fused scaling + walk-migration comm only — still a valid result.
