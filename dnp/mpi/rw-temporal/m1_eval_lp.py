#!/usr/bin/env python3
"""
m1_eval_lp.py -- temporal link-prediction eval for the in-engine co-shard
embeddings (M2 increment 1). Loads the embedding shards the engine dumped,
does the SAME chronological 80/20 split as e2_negsample.py, and reports
AUC / AP on held-out future edges, so the in-engine number is comparable to
E2's single-machine numbers.

Usage: m1_eval_lp.py <full_edge_file> <embed_glob> [split]
  e.g. m1_eval_lp.py data/wikipedia.txt 'log/embed_wikipedia_train_p4_r*.txt'
"""
import glob, sys
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0] if '/' in __file__ else '.')
from partition_metis import load_edges


def main():
    edge_file = sys.argv[1]
    embed_glob = sys.argv[2]
    split = float(sys.argv[3]) if len(sys.argv) > 3 else 0.8

    edges = load_edges(edge_file)
    edges = edges[np.argsort(edges[:, 2], kind='stable')]
    cut = int(split * len(edges))
    test = edges[cut:, :2]

    # load embeddings (global_id -> vector) from all shard files
    emb = {}
    files = sorted(glob.glob(embed_glob))
    if not files:
        sys.exit(f"no embedding shards match {embed_glob}")
    d = None
    for f in files:
        for line in open(f):
            p = line.split()
            gid = int(p[0]); vec = np.asarray(p[1:], dtype=np.float32)
            emb[gid] = vec
            if d is None: d = len(vec)
    print(f"loaded {len(emb)} embeddings (dim={d}) from {len(files)} shards")

    # keep test edges whose both endpoints have embeddings
    keep = [(u, v) for u, v in test if u in emb and v in emb]
    if not keep:
        sys.exit("no test edges with both endpoints embedded")
    keep = np.array(keep)
    ids = np.array(list(emb.keys()))
    rng = np.random.default_rng(7)

    def vec(arr):
        return np.stack([emb[int(x)] for x in arr])

    u = keep[:, 0]; v = keep[:, 1]
    vneg = rng.choice(ids, size=len(u))
    s_pos = np.sum(vec(u) * vec(v), axis=1)
    s_neg = np.sum(vec(u) * vec(vneg), axis=1)

    scores = np.concatenate([s_pos, s_neg])
    labels = np.concatenate([np.ones(len(s_pos)), np.zeros(len(s_neg))])
    order = np.argsort(scores); ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    npos, nneg = len(s_pos), len(s_neg)
    auc = (ranks[:npos].sum() - npos * (npos + 1) / 2) / (npos * nneg)
    o = np.argsort(-scores); lab = labels[o]
    tp = np.cumsum(lab); prec = tp / np.arange(1, len(lab) + 1)
    ap = (prec * lab).sum() / max(1, lab.sum())
    print(f"temporal LP: AUC={auc:.4f}  AP={ap:.4f}  "
          f"(test_edges_used={len(keep)})")


if __name__ == "__main__":
    main()
