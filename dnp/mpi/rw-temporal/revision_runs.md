# HPCC 2026 — revision run batch (data the paper still needs)

## STATUS (2026-06-22)
- **#3 ρ=0 vs global — DONE & in paper.** 3 seeds × {wikipedia, reddit, tgbl_review},
  METIS, d=64. Result: local/global MRR = wikipedia **102%**, reddit **97%**,
  tgbl-review **87%**. The old "96–97%" claim was an overclaim → corrected to
  "87–102%" in abstract/§III-C/§V-F + new Table V.
- **T4 pilot timing (local sanity) — DONE & in paper.** 20k-walk pass: reddit 2.34s,
  wikipedia 0.99s → §III-D now says "<2.5s on the quality graphs, paid once."
  Representative stackoverflow pilot timing still wanted on Wisteria.
- **#4 quality-at-scale — DONE & in paper (§V-F).** 384-rank (8-node) fused run on
  tgbl_review_train, embed_dump → m1_eval_lp on full file: **historical MRR 0.157**
  (within the single-machine 0.156–0.160 range) despite a single online pass →
  quality holds at cluster scale. NOTE: `-x VAR,VAR2` forward-by-name silently
  failed (ran script defaults); use `-x VAR=value`. Cluster numpy 3.6 needs
  `sed -i "s/kind='stable'/kind='mergesort'/g" m1_eval_lp.py` (don't commit).
- **#1 overlap baseline — DONE via analysis (§V-B), not code.** Instead of risky
  untested async MPI (MPI_Ialltoallv; existing two-stage already hit a deadlock),
  added an overlap BOUND: with perfect overlap two-stage time ≥ max(compute,
  emb_xchg); emb_xchg 3.1–6.9s exceeds compute 0.73–0.86s by 4–9×, so overlap
  hides at most one compute phase → two-stage stays exchange-bound at every scale.
  (Empirical async baseline still possible later if a reviewer insists.)
- **T4 pilot % — DONE & in paper (§III-D).** Ran 500k walks on the full 17.8M-edge
  stackoverflow via .part chunks (.venv_part, load_edges is chunk-aware): **100.5s
  real**, one-time offline. Corrected the earlier "dwarfed by per-round training"
  wording (wrong: 100s > a 1.8–6.7s run) → now "one-time preprocessing, reused
  across every run/scale/rep." NOTE the cluster `python3 ... data/stackoverflow_a2q.txt`
  gave no output because the base/.part isn't on /work (only data/384/ uploaded);
  pilot is offline so running it on the Mac chunks is fine.

---

Three measurements remain before the HPCC draft is fully defensible. All env
vars / scripts below are the **real** ones in this repo (verified against
`main.c`, `e2_negsample.py`, `job_scaling.sh`). Hand the resulting numbers back
and the paper tables/figures get filled in.

Ordered by value ÷ effort. **#3 and the T4-pilot timing are cheap and single-node;
#4 needs a cluster job; #1 needs a small code change and is optional.**

---

## #3 — ρ=0 (shard-local) vs global negatives, per dataset  ⟵ do first, cheapest

**Why:** §III-C and §V-F assert "ρ=0 reaches 96–97% of global-negative MRR" but
the paper shows **no data**. This is the load-bearing justification for dropping
the exchange machinery — it needs its own table.

**Tool:** `e2_negsample.py` (single-machine, NO MPI). CLI:
`python e2_negsample.py <edge_file> <n_shards> <dim> <epochs>`
Regimes via `E2_REGIMES`: `global` (reference upper bound) vs `local` (ρ=0, pure
shard-local). **Must use METIS shards** (`E2_SHARDS=metis`, the default) — the
96–97% result is METIS-specific; random shards give the old, worse number.

**Run** (3 seeds × 3 datasets, regimes global+local, 4 METIS shards, d=64).
CLI is `e2_negsample.py <edge_file> <n_shards> <dim> <epochs>` — **no P arg**.
Use **`.venv_part/bin/python`** (needs `pymetis`, only in that venv). wikipedia +
reddit have local base files; **tgbl_review base is only on /work** (local has
just `.part` chunks) — copy it over first.
```bash
cd dnp/mpi/rw-temporal
for ds in wikipedia reddit tgbl_review; do      # tgbl_review needs /work base
  for s in 0 1 2; do
    E2_SHARDS=metis E2_REGIMES=global,local E2_SEED=$s \
      .venv_part/bin/python e2_negsample.py data/${ds}.txt 4 64 3 \
      | tee log/e2_rho0_${ds}_s${s}.txt
  done
done
```
**Collect:** per dataset, the `global` MRR and `local` MRR (mean over 3 seeds) and
the ratio local/global. **Paper target:** new small table in §V-F (or a 4th bar
group in Fig 5) — e.g. `wikipedia: local 0.xx / global 0.yy (zz%)`.

---

## T4 pilot-pass cost (the "cheap" claim in §III-D)  ⟵ also cheap

**Why:** §III-D calls the pilot walk pass "cheap" with no number; reviewer #9
asked to quantify it.

**Tool:** `pilot_edge_weights.py` — CLI is `<edge_file> [n_walks] [max_steps]`
(**arg 2 is n_walks, NOT a part count**; leave default 20000 to match the real
pilot). It writes the `.ew` weights; the timing goes to stderr. Run on **Wisteria**
against `stackoverflow_a2q` for a representative number (its base file is on
`/work`, not local). macOS `/usr/bin/time` has **no `-v`** — drop it on the Mac.
```bash
# Wisteria (GNU time supports -v); stackoverflow base lives on /work
/usr/bin/time -v .venv_part/bin/python pilot_edge_weights.py /work/<path>/stackoverflow_a2q.txt \
  2> log/t4_pilot_time.txt
# Mac sanity only (no -v; stackoverflow base not local, use reddit)
/usr/bin/time .venv_part/bin/python pilot_edge_weights.py data/reddit.txt 2> log/t4_pilot_reddit.txt
```
**Collect:** pilot wall-time as a % of one full training run's elapsed (use the
strong-scaling elapsed at the same P). **Paper target:** one clause in §III-D,
e.g. "the pilot pass costs ≈X% of a single run, paid once."

---

## #4 — embedding quality at cluster scale  ⟵ needs a multi-node job

**Why:** all quality numbers are single-machine; the scaling runs are on the
cluster. The paper never shows the embedding it *ships at scale* is good.
`stackoverflow_a2q` LP is degenerate (AUC≈0.55), so use **`tgbl-review`** (large,
4.87M edges, has LP signal) for this.

Must train on a **train-only 80% split** (else test edges leak); `m1_eval_lp.py`
re-derives the same 80/20 time split from the FULL file and tests on the held-out
20% — mirrors the inc-1 methodology. `embed_dump` writes original global ids, so
train-partition shards and full-file test ids match. P=384 (8 nodes) is the
representative point; TOTAL=2M walkers matches the de-risk coverage.

**A. On the Mac** (`.venv_part` has pymetis) — partition, upload:
```bash
cd /Users/smallcat/Documents/GitHub/graph/dnp/mpi/rw-temporal
# A1 SKIP — data/tgbl_review_train.txt already exists & validated (Jun 14):
#    3,898,832 lines == int(0.8*4,873,540), timestamps monotonic (temporal prefix),
#    the same file that produced the de-risk hist-MRR 0.156-0.160. Reuse it.
# A2 partition the existing TRAIN file into 384 parts (static METIS)
.venv_part/bin/python partition_metis.py data/tgbl_review_train.txt 384
# A3 upload the partition dir (full tgbl_review.txt must already be on /work for step C)
rsync -av data/384 \
  z30130@wisteria.cc.u-tokyo.ac.jp:/work/gz00/z30130/rw-temporal/data/
```
**Step C needs the FULL `data/tgbl_review.txt` on /work** (not the train file) — it
re-derives the held-out 20%. The Mac has only `.part` chunks + the train file now;
ensure the full file is on /work (it was readable for #3), or `cat
data/tgbl_review.txt.part* > data/tgbl_review.txt` if those are raw edge chunks.

**B. On Wisteria login node** — build, then submit the fused run:
```bash
cd /work/gz00/z30130/rw-temporal       # your /work staging dir
sh wisteria/build_wisteria.sh          # if ./rw not built yet
mkdir -p log                           # embed_dump writes here (LOG_DIR="log")
# Use -x VAR=value (with '='), NOT -x VAR,VAR2 (forward-by-name silently failed
# once -> job ran the script DEFAULTS stackoverflow_a2q_tw/twostage instead).
pjsub -x DATASET=tgbl_review_train,EMB=1,EMBMODE=fused,TOTAL=2000000 -g gz00 \
      -L node=8 --mpi proc=384 wisteria/job_scaling.sh
# -> writes log/embed_tgbl_review_train_p384_r*.txt
# EMBMODE=fused is required: it trains the shipped local-negative embeddings;
# twostage trains the global-negative variant (not what #4 evaluates).
# Fallback if -x still won't forward: edit the DATASET/EMBMODE defaults in
# job_scaling.sh directly, then pjsub with just -L/--mpi.
```

**C. Offline eval** (login node, python with numpy — the README `venv`):
```bash
cd /work/gz00/z30130/rw-temporal
venv/bin/python m1_eval_lp.py data/tgbl_review.txt \
  'log/embed_tgbl_review_train_p384_r*.txt' 0.8
```
Prints global / time-local / **historical** MRR. The **historical** number is the
paper figure. **Sanity first (optional, cheap):** run the same A2/B/C at `P=4`
with `mpiexec -n 4 ./rw tgbl_review_train ...` locally to confirm shard ids match
the eval before spending the 8-node job.

**Caveat to report:** the cluster run is a **single online pass** (vs the 3-epoch
single-machine de-risk that gave hist-MRR 0.156–0.160), so expect a somewhat lower
absolute number; the point is parity/consistency at scale, not a new high.
**Paper target:** one row/sentence in §V-F closing the quality↔scale gap.

---

## #1 — overlap-enabled two-stage baseline  ⟵ OPTIONAL (needs code), strengthens headline

**Why:** the headline's +38–154% is vs a **blocking** two-stage exchange. The
paper now honestly frames the *eliminated volume* as the invariant (§V-B, §III-F),
so this is no longer a blocker — but an overlap baseline would convert "could
narrow the gap" into a measured residual.

**Code:** add `EMBED_MODE=twostage_overlap` in `main.c` / `embed.c`: replace the
blocking `MPI_Alltoallv` in `twostage_embed_exchange()` with `MPI_Ialltoallv`
issued *before* the local compute of the next round, then `MPI_Wait` after. Time
the *non-overlapped* residual as `emb_xchg`.

**Run:** the headline sweep again with the new mode:
```bash
for P in 96 192 384 768 1536; do
  DATASET=stackoverflow_a2q EMB=1 EMBMODE=twostage_overlap pjsub wisteria/job_scaling.sh
done
```
**Collect:** emb_xchg / elapsed for `twostage_overlap` vs `fused` vs `twostage`.
**Paper target:** a third series in Fig 3 / Table III — shows overlap shrinks but
does not erase the gap.

---

### Hand-back
Drop the `log/*.txt` (or just the summarized numbers) and I'll: build the §V-F
ρ=0 table, add the T4 pilot clause, add the cluster-quality row, and (if #1 is run)
extend the headline figure. `analyze_scaling.py` already parses `*.out` →
median[min–max] per (dataset,mode,proc) for the scaling-style outputs.
