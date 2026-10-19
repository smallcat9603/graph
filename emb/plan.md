# emb — plan

Edited in place: this file shows the current state, not its history.
Settled discussion goes to `measurement_log.md`; this file cites log entries by number.

## Current state

- **Done:** scaffold; refactored copy of `dnp/kg` (`embkg/` library, `app/` viewer, `scripts/rwemb.py`, regression tests; log #1). Database-free path verified (log #2).
- **Open:** Neo4j smoke test of the app; decisions D1–D3 below; research question; target venue; artifact folder.
- **Latest log entry:** #2.

## To do

### Decisions for the author

- **D1 — make eigsh reproducible** (log #2, finding B). Options:
  1. Pass a fixed start vector `v0` to `eigsh`. Topology-only results stay within ~1e-6; label-mixed results become deterministic but differ from the saved files. *(Recommended.)*
  2. Keep the random start, and report results as mean ± std over several embeddings.
- **D2 — non-symmetric R under label mixing** (log #2, finding B). Options:
  1. Symmetrize, R ← (R + Rᵀ)/2, before `eigsh`. Smallest change, keeps the method's form. *(Recommended, together with D1.1.)*
  2. Use an SVD (`svds`) of R, which does not assume symmetry.
  3. Keep as is and state the limitation.
  Any option changes the label-mixed numbers behind ADMA 2025 / Innovation Discovery; those papers are not edited, but a new paper must use the new numbers.
- **D3 — verify finding A** (cached read-back in `dnp/kg` tuning; log #2). Rerun node2vec tuning on newfood in `dnp/kg` and in `emb` with the same trial budget. Needs Neo4j and APOC Extended. Decide whether this matters for any claim you still make.

### Design decisions

- Research question — **author to decide.** Candidate directions, from the published papers' stated limitations (log #1) and from rw-temporal:
  - automatic or node-adaptive α/β instead of hand tuning;
  - heterogeneous or time-varying layers (the memoryless linear mixing does not cover them), possibly with the rw-temporal walk engine;
  - scaling RWEM beyond dense n×n matrices (sparse or distributed walks, as in rw-temporal);
  - the missing baselines (GraRep/HOPE, MNE/PMNE, R-GCN/HGT) on the multiplex graph.
- Target venue, page limit and deadline — **author to decide.**
- Environment runbook (spec §6) — local Mac + local Neo4j Desktop for now; write it out once the first experiment is designed.

### Code

- `embkg/rwemb.py` is dense (n×n `matrix_power`, `eigsh`): fine for airport (3,158 nodes, ~5 s), not for large graphs.
- APOC Extended is missing from the local DBMS (log #1); install it before URL loading, Online mode or the CYPHER datasets.

### Artifact structure and page budget

Not started.

### Open questions

- Should a new paper correct the ADMA paper↔code mismatches (mixing order, β notation; log #1), or only describe the method as implemented?

## Results

One entry per artifact id (`fig:x` / `tab:x` / `§x`), each with a claim block (spec §4) and a reproduction block (spec §2).
No entries yet.

<!-- Entry template
### fig:x — <short name> (§N)

**Claim.** <claim, bound inside the sentence> · **Backing:** log #N · **Strength:** STRONG | BOUNDARY · **Reviewer will probe:** <...>

**Reproduction.**
- Environment: <link to runbook>
- Raw numbers: log #N
- Data-gen: `<exact command>`
- Generator: `<script>` — PARSES | HARDCODES → `<output path>`, wired at `fig:x`
-->
