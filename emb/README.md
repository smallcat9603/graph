# emb

Node-embedding research on knowledge graphs, continuing `graph/dnp/kg`
(published as ADMA 2025, APCC 2025 and Innovation Discovery 2026).
`dnp/kg` is left untouched; this folder holds the refactored copy.

## Layout

| Path | What |
|---|---|
| `embkg/` | Library, no Streamlit: `rwemb` (random-walk embedding), `evaluate` (SVM, t-SNE), `gds_ops` (FastRP, node2vec, kNN, node similarity), `cypher_ops` (graph construction queries), `datasets`, `tuning` (Optuna), `paths` |
| `app/` | Streamlit + Neo4j viewer. `Home.py` connects; `pages/` as in `dnp/kg`; `ui/` holds the shared widgets and cached wrappers |
| `scripts/rwemb.py` | Headless RWEM run: embedding file + SVM accuracy as JSON in `out/emb/` |
| `tests/` | Regression against the original `dnp/kg` code and its saved embeddings |
| `out/` | Outputs (not in git) |

Data is read from its original place (`graph/dnp/kg/data`, `graph/data`, `graph/cypher`); Neo4j reads the same files from GitHub (see `embkg/paths.py`).

## Setup

```bash
/opt/homebrew/bin/python3.13 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m spacy download en_core_web_sm
.venv/bin/python -m spacy download ja_core_news_sm
```

Neo4j credentials come from `~/.streamlit/secrets.toml` (`NEO4J_LOCAL`, `NEO4J_LOCAL_USER`, `NEO4J_LOCAL_PASSWORD`); set `EMB_NEO4J=SANDBOX` to use the sandbox entries.
The DBMS needs the GDS and APOC plugins.
URL loading, `apoc.load.html`, the GCP NLP mode and the CYPHER datasets (`apoc.cypher.runFile`) also need APOC Extended.

## Run

```bash
.venv/bin/streamlit run app/Home.py
```

```bash
.venv/bin/python scripts/rwemb.py --edges ../data/airport_0.edges --labels ../data/airport.continent_0.labels --labels2 ../data/airport.country_0.labels --alpha 0.5 --beta 0.5 --seed 0
```

```bash
.venv/bin/python -m pytest -q
```
