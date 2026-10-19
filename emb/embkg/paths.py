"""Locations of code, data and outputs.

Data stays where the original project keeps it (read only):
- graph/dnp/kg/data   knowledge-graph CSVs and DNP news releases
- graph/data          edge lists and label files (airport, blogcatalog, ...)
Neo4j cannot read local paths outside its import folder, so it reads the same
files from GitHub (main branch) through RAW_BASE.
New outputs go to emb/out/, which is not tracked by git.
"""

from pathlib import Path

EMB_ROOT = Path(__file__).resolve().parents[1]
GRAPH_ROOT = EMB_ROOT.parent

KG_DIR = GRAPH_ROOT / "dnp" / "kg"
KG_DATA_DIR = KG_DIR / "data"
NEWSRELEASE_DIR = KG_DATA_DIR / "newsrelease_B-1-100_C-1-4"
GRAPH_DATA_DIR = GRAPH_ROOT / "data"

RAW_BASE = "https://raw.githubusercontent.com/smallcat9603/graph/main/"
KG_DATA_URL = RAW_BASE + "dnp/kg/data/"
CYPHER_URL = RAW_BASE + "cypher/"
GRAPH_DATA_URL = RAW_BASE + "data/"

OUT_DIR = EMB_ROOT / "out"
EMB_OUT_DIR = OUT_DIR / "emb"
