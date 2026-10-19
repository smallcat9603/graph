# emb — measurement log

Append only: never edit an entry after it is written.
The one exception: when a result later feeds an exhibit, add a one-line pointer to the exhibit's id at the end of its entry.
Number entries consecutively (#1, #2, …).
Each entry records the date, the exact command, script and environment, the raw numbers, the finding, the boundary, and any exclusion with its reason and timestamp (spec §1, §8).
For a decision experiment, write the interpretation rule in an entry before the data exists.

## #1 — 2026-10-19 — Project origin and refactor of dnp/kg

**Positioning (author, 2026-10-19).** This project continues `graph/dnp/kg`, mainly by extending its node-embedding work.
Ideas may also be combined with `graph/dnp/mpi/rw-temporal` (HPCC 2026).
The Streamlit + Neo4j app stays the viewer for code progress and outputs.
`dnp/kg` itself must not be modified.

**Prior publications built on dnp/kg** (sources under `$DB` = the author's Dropbox `胡曜/`):
- ADMA 2025, "RWEM: An In-memory Random Walk Based Node Embedding Framework on Multiplex User-Item Graphs" — `$DB/2026/Innovation_Discovery/adma2024/main.tex`.
  Uses `pages/50_RW_Emb.py` → `flow.node_emb` / `M_attr` on OpenFlights airport (3,158 nodes, 18,606 edges; continent and country labels; 20%-edge-removed variant). Baselines FastRP and node2vec via GDS.
- APCC 2025, "A Parallelized GDB-Driven Framework for Scalable Knowledge Reasoning" — `$DB/2025/apcc2025(accept)/main.tex`.
  Uses graph construction (four load modes), node similarity, FastRP/node2vec/RWEM and Optuna tuning (`objective_n2v` on the newfood graph).
- Innovation Discovery 2026, "From Topology to Cognition: …" — `$DB/2026/Innovation_Discovery/ID20260046_compressed_hu.pdf`.
  Combines ICICN 2023, ADMA and APCC. Its Sec. 5 lists as limitations: hand-tuned α/β; the layer reduction assumes one shared node set and memoryless linear mixing; no comparison with R-GCN/HGT, Laplacian eigenmaps, GraRep/HOPE, MNE/PMNE/DANE, DyRep/TGAT/TGN.

**Paper ↔ code mismatches noticed while reading** (not yet raised in any paper):
- ADMA describes mixing adjacency matrices and then row-normalizing; `M_attr` mixes transition matrices that are already row-normalized per layer.
- In code `beta` splits the label share between label 1 and label 2; the paper's β values (0.5/0.25/0.05) correspond to (1−α)·β, and the saved files use beta 1.0/0.5/0.1.

**Refactor decisions (author, 2026-10-19).**
- Carry over all app features: RW and GDS embedding, graph construction and node similarity, Chart page, Bot page. Old versions `neo4j_v0`–`v5`, notebooks and `streamlit/` demos are not carried over.
- Data stays where it is, read only: `graph/dnp/kg/data`, `graph/data`, `graph/cypher` (Neo4j reads them from GitHub raw URLs). New outputs go to `emb/out/` (git-ignored).
- New environment `emb/.venv`: Homebrew Python 3.13.14; versions pinned in `requirements.txt` (streamlit 1.66.0, graphdatascience 2.2, numpy 2.5.4, scipy 1.18.1, igraph 1.0.0, networkx 3.7, optuna 5.0.0, spacy 3.8.16, langchain-neo4j 0.11.0).
- Local Neo4j DBMS (Desktop 2.1.3, server 2026.02.2) has GDS 2.27.0 and APOC core 2026.02.2 only. APOC Extended (`apoc.load.html`, `apoc.nlp.gcp.*`, `apoc.cypher.runFile`) is absent, so URL loading, the Online mode and the CYPHER datasets will fail until it is installed.

**Structure.** `embkg/` library without Streamlit; `app/` Streamlit pages over it; `scripts/rwemb.py` headless runs; `tests/` regression. See `README.md`.

**Behaviour changes against dnp/kg** (everything else ported unchanged):
- Embeddings are read back from Neo4j uncached (see #2, finding A).
- The RW page's cache is keyed by the edge-file name as well, since the graph object is not hashed.
- Optuna plots are drawn in the page (`st.plotly_chart`) instead of `fig.show()`, which opened a separate browser tab.
- The Bot page calls `GraphCypherQAChain` directly (LangChain 1.x has no `initialize_agent`); the 5-turn chat memory is gone.
- `evaluate.svm_accuracy` takes an optional `seed`; the app keeps the original unseeded splits.
- Saved RW embeddings go to `emb/out/emb/` instead of `dnp/kg/emb/`.

## #2 — 2026-10-19 — Regression check of the refactor, and two reproducibility findings

**Environment.** MacBook (darwin), `emb/.venv` as in #1. No Neo4j running, so only the database-free path was run.

**Regression.** `.venv/bin/python -m pytest -q` → 9 passed (68.5 s).
- Transition matrix (igraph and networkx), `M_attr` at (α,β) = (0.5,0.5), (0.5,0.1), (0.8,1.0), autocovariance and PMI at τ=3: bit-identical to `dnp/kg/pages/lib/flow.py`.
- Topology-only embedding (airport_0, igraph, τ=3, dim 128) vs the saved `dnp/kg/emb/airport_0_tau3_dim128_alpha1.0_beta1.0.txt`: max abs difference 0.0 in one run, 1.38e-6 in others. So the saved file was made with igraph, not networkx (networkx: column match median |cos| 0.37).

**Finding A — cached read-back in dnp/kg (code reading; not yet confirmed by a run).**
`cypher.get_emb_result` runs its query through `run()`, which is `@st.cache_data` keyed by the query text.
The text does not change when the embedding is recomputed.
So within one session, `objective_frp` / `objective_n2v` would score every Optuna trial on the first trial's embedding, and "Plot embeddings" would show a stale embedding.
Trial-to-trial differences would then come only from the unseeded SVM splits.
This bears on the APCC 2025 tuning result (node2vec on newfood, mean accuracy 0.62 → 0.71, parameter importances).
To confirm, rerun the tuning with Neo4j in both versions (an experiment for the author to approve).

**Finding B — eigsh is not reproducible with label mixing.**
- `eigsh` starts from a random vector (scipy default `v0`).
- With labels, the clean autocovariance R is not symmetric (max |R−Rᵀ| = 2.73e-5), although `eigsh` assumes it is.
  Its eigenvalues near k=128 are about 2.2e-5, with gaps of about 1e-7.
- Two runs in one process, identical input (airport_0, continent+country, α=β=0.5, τ=3, dim 128): max abs difference 70.6 after rescaling.
- Same input with a fixed `v0 = ones(n)`: difference 0.0.
- vs the saved `dnp/kg/emb/airport_0_tau3_dim128_alpha0.5_beta0.5.txt`: no label-file combination matches (max abs 48–104).
  Subspace angles to our run: median 2.88°, max 44.1°. The leading directions agree; the tail rotates.
- Topology only (R symmetric up to rounding, well-separated spectrum) is stable to ~1e-6.
- Boundary: one dataset, one (τ, dim). Effect on downstream accuracy not yet measured.

**CLI smoke run.**
`.venv/bin/python scripts/rwemb.py --edges ../data/airport_0.edges --labels ../data/airport.continent_0.labels --labels2 ../data/airport.country_0.labels --alpha 0.5 --beta 0.5 --seed 0`
→ `out/emb/airport_0_tau3_dim128_alpha0.5_beta0.5.{txt,json}`; 3,158 nodes; embed 5.26 s; SVM accuracy over 5 seeded splits [0.997468, 0.997468, 0.996203, 0.997468, 0.997468], mean 0.997215, std 0.000506.
One embedding only (eigsh unseeded, finding B), so this is not a measurement of variance across embeddings.

**App check without Neo4j.** Streamlit `AppTest` ran the DATA, RW_Emb, Chart and Changelog pages with no exception. Pages that need Neo4j are not yet checked.
