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

    # dense remap: global id -> row in W
    id_list = list(emb.keys())
    idx_of = {g: i for i, g in enumerate(id_list)}
    W = np.stack([emb[g] for g in id_list]).astype(np.float32)
    dense = np.array([(idx_of[int(u)], idx_of[int(v)]) for u, v in keep])

    # bipartite? -> destination negatives from the dst-side embedded nodes only
    src_set = set(int(x) for x in edges[:, 0]); dst_set = set(int(x) for x in edges[:, 1])
    if len(src_set & dst_set) <= 0.02 * min(len(src_set), len(dst_set)):
        dst_pool = np.array([idx_of[g] for g in dst_set if g in idx_of], dtype=np.int64)
        kind = f"bipartite (|dst|={len(dst_pool)})"
    else:
        dst_pool = None
        kind = "general (all-node negs)"

    from e2_negsample import eval_lp
    rng = np.random.default_rng(7)
    auc, mrr, h10 = eval_lp(W, dense, len(id_list), rng, dst_pool=dst_pool)
    print(f"temporal LP [{kind}]: AUC={auc:.4f}  MRR={mrr:.4f}  Hits@10={h10:.4f}  "
          f"(test_edges_used={len(keep)})")


if __name__ == "__main__":
    main()
