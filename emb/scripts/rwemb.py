"""Headless RWEM run: edge list (+ labels) -> embedding file + SVM accuracy.

Same computation as the RW_Emb page, without Streamlit or Neo4j, so runs can be
scripted and logged. Example (airport, as in ADMA 2025):

    .venv/bin/python scripts/rwemb.py --edges ../data/airport_0.edges \
        --labels ../data/airport.continent_0.labels --labels2 ../data/airport.country_0.labels \
        --alpha 0.5 --beta 0.5 --seed 0
"""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np

from embkg import evaluate, graphio, paths, rwemb


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--edges", required=True)
    p.add_argument("--labels")
    p.add_argument("--labels2")
    p.add_argument("--graph-tool", default="igraph", choices=["igraph", "networkx"])
    p.add_argument("--sim", default="Autocovariance", choices=["Autocovariance", "PMI"])
    p.add_argument("--tau", type=int, default=3)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--alpha", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=None, help="seed for the SVM splits (default: unseeded, as in the app)")
    p.add_argument("--out-dir", default=str(paths.EMB_OUT_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    t0 = time.perf_counter()

    G = graphio.build_graph(graphio.read_edges(args.edges), args.graph_tool)
    u, category = rwemb.node_emb(
        G, sim=args.sim, tau=args.tau, dim=args.dim, graph_tool=args.graph_tool,
        df_label=graphio.read_labels(args.labels), df_label2=graphio.read_labels(args.labels2),
        alpha=args.alpha, beta=args.beta,
    )
    t_emb = time.perf_counter() - t0

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    name = Path(args.edges).stem
    stem = f"{name}_tau{args.tau}_dim{args.dim}_alpha{args.alpha}_beta{args.beta}"
    np.savetxt(out_dir / f"{stem}.txt", u)

    record = {
        "argv": sys.argv,
        "python": platform.python_version(),
        "n_nodes": int(u.shape[0]),
        "embed_seconds": round(t_emb, 3),
        "embedding": str(out_dir / f"{stem}.txt"),
    }
    if len(np.unique(category)) > 1:
        acc_scores = evaluate.svm_accuracy(evaluate.emb_frame(u, category), seed=args.seed)[0]
        record["svm_acc"] = [round(float(a), 6) for a in acc_scores]
        record["svm_acc_mean"] = round(float(np.mean(acc_scores)), 6)
        record["svm_acc_std"] = round(float(np.std(acc_scores)), 6)

    (out_dir / f"{stem}.json").write_text(json.dumps(record, indent=1))
    print(json.dumps(record))


if __name__ == "__main__":
    main()
