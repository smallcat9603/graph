# Next-paper experiment plan — partition-count scaling of embedding quality

Created 2026-09-22, carved out of the HPCC-2026 camera-ready scope.
The HPCC paper (#1571315639, accepted) defers these experiments; its §VI
discloses the gap. This file is the plan for the follow-on paper.

## Why this is the right next paper

Two of the three HPCC reviewers converged on the same hole, independently:

- **R2:** *"The main quality study for shard-local negatives is conducted with
  four shards, whereas the scaling experiments reach 1536 ranks. It would be
  very helpful to show how embedding quality changes as the number of partitions
  increases, even on a smaller dataset where 4-, 8-, 16-, and 32-way tests are
  practical. This would clarify whether local-negative bias and dropped
  cross-partition pairs remain acceptable at higher degrees of fragmentation."*
- **R2, again:** *"Please add a sensitivity analysis relating cross-partition-pair
  rate to link-prediction quality."*
- **R3:** *"shard-local negatives introduce a non-negligible quality loss on
  tgbl-review (about 13% MRR), so the communication-quality trade-off deserves
  slightly more analysis."*

That is a reviewer-validated research question, already framed, for a design the
HPCC paper establishes works. It is a strong opening for the follow-on.

## The gap in one sentence

The HPCC paper shows shard-local negatives recover 87–102% of global-negative
quality **at 4 shards**, and shows the system scales to **1536 ranks** — but
never connects the two axes. Nobody has shown what happens to local-negative
quality at the fragmentation levels the engine actually runs at.

## Experiment design

**No code changes needed.** Both scripts already take the shard count as
`argv[2]`, verified 2026-09-22:

```
.venv_part/bin/python e2_negsample.py   data/<ds>_train.txt <P> 64 3
.venv_part/bin/python inc2_crosspair.py data/<ds>_train.txt <P> 64 3
```

| | E1 — local vs global negatives | E2 — cross-pair rate vs quality |
|---|---|---|
| script | `e2_negsample.py` | `inc2_crosspair.py` |
| sweep | P ∈ {4, 8, 16, 32, 64, 128} | same |
| datasets | wikipedia, reddit, tgbl-review | same |
| seeds | 3 | 3 |
| metric | local/global MRR ratio vs P | keep−drop AP gap vs P, against the measured cross-pair rate |

Extending the sweep past 32 (to 64/128) is worth it and cheap — the HPCC
engine runs at 1536 ranks, so the more fragmentation the curve covers, the more
it actually answers R2's question rather than gesturing at it.

**Measured cost** (this Mac, 2026-09-22): wikipedia @ P=8 took **50 s** wall.
reddit is ~5× the edges (~4 min/point). tgbl-review is 93 MB, ~45× wikipedia
(~40 min/point). wikipedia + reddit full sweep ≈ 2–3 h; tgbl-review overnight.

**Run E2 on the same partitions as E1** so the two panels share an x-axis and
the cross-pair rate is the measured one, not a modelled one.

## Recorded probe — wikipedia, P=8, single seed (2026-09-22)

Not a result; a feasibility check that the sweep runs and produces sane output.

```
./.venv_part/bin/python e2_negsample.py data/wikipedia_train.txt 8 64 3
nodes=8007 edges=125979 train=100783 test=25196 shards=8 dim=64 ep=3
walks=35430 positive_pairs=1195554
METIS shard sizes [1001 1001 1000 1001 1001 1002 1001 1000]
eval: bipartite (type-aware negs, |dst|=973)

regime          AUC     MRR    H@10
global       0.7434  0.5179  0.6153
local        0.8444  0.5500  0.6628
local_iw     0.8146  0.5136  0.6252
local_iw_x   0.8001  0.5156  0.6180
wall: 50.4 s
```

Two observations, both preliminary:

1. **local/global MRR = 1.062 at P=8**, i.e. no degradation versus the 1.02
   the paper reports at P=4. Weak evidence that the recovery ratio does not
   fall off immediately with fragmentation.
2. **The importance-weighted correction is worse than plain local**
   (`local_iw` 0.5136 vs `local` 0.5500), independently corroborating the
   HPCC paper's decision to drop the correction.

**Boundary — do not cite these numbers.** Single seed, not the 3-seed protocol
behind the paper's Table V. The absolute values are not comparable to Table V
either, which is 4 shards and a 3-seed mean. Only the within-run ratio is
meaningful, and one seed cannot support it.

## What the follow-on paper could claim

Depends on which way the curve goes, and both directions are publishable:

- **If the ratio holds to P=128:** shard-local negatives are a general
  communication-elimination technique, not a small-partition artifact. Strongest
  outcome; upgrades the HPCC design from "works at our scale" to "works".
- **If it degrades past some P\*:** then P\* is a real, measurable design
  boundary, and the paper becomes about *when* you must pay for global
  negatives — which motivates the ρ>0 correction the HPCC paper dropped, now
  with evidence for when it earns its cost. Arguably the more interesting paper.

Either way, pair it with the placement study R2 also asked for (TRP's
non-monotonic wall-clock vs Tofu rank mapping) if Wisteria time allows — that
is the other deferred HPCC item and it belongs to the same follow-on.

## Rigor rules carried over

- 3 seeds minimum; report mean±std with n, not bare means (HPCC R1: *"please use
  statistical confidence when plotting results — the mean can be deceiving"*).
- Hold walks, positive pairs, init and RNG identical across regimes so the only
  variable is how negatives are drawn — `e2_negsample.py` already does this.
- Record every run the same session it is produced, with the exact command.
