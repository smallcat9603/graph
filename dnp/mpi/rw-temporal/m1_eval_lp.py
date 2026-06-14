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
    test_t = edges[cut:]          # (M,3) u v t  -- keep time for time-local eval
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
    print(f"GLOBAL-neg LP [{kind}]: AUC={auc:.4f}  MRR={mrr:.4f}  Hits@10={h10:.4f}  "
          f"(test_edges_used={len(keep)})")

    # --- time-local eval: negatives drawn from destinations active in the SAME
    #     time window as the positive (temporally-contemporaneous). Tests
    #     fine-grained temporal discrimination that global-neg LP hides.
    keep_mask = np.array([(int(u) in idx_of and int(v) in idx_of) for u, v in test])
    tt = test_t[keep_mask]
    u_d = np.array([idx_of[int(x)] for x in tt[:, 0]])
    v_d = np.array([idx_of[int(x)] for x in tt[:, 1]])
    times = tt[:, 2].astype(np.float64)
    NB = 20
    lo, hi = times.min(), times.max()
    binof = np.clip(((times - lo) / (hi - lo + 1e-9) * NB).astype(int), 0, NB - 1)
    Kn = 100
    inv_rank = []; hits = []
    for b in range(NB):
        m = np.where(binof == b)[0]
        if len(m) < 2:
            continue
        pool = np.unique(v_d[m])            # dst active in this window
        if len(pool) < 2:
            continue
        ub = u_d[m]; vb = v_d[m]
        pos = np.sum(W[ub] * W[vb], axis=1)
        negi = pool[rng.integers(len(pool), size=(len(m), Kn))]
        neg = np.einsum('pd,pkd->pk', W[ub], W[negi])
        ge = (neg >= pos[:, None]).sum(axis=1)
        inv_rank.append(1.0 / (ge + 1)); hits.append(ge < 10)
    if inv_rank:
        mrr_t = float(np.concatenate(inv_rank).mean())
        h10_t = float(np.concatenate(hits).mean())
        print(f"TIME-LOCAL-neg LP: MRR={mrr_t:.4f}  Hits@10={h10_t:.4f}  "
              f"(negs = destinations active in same time window, {NB} bins)")

    # --- HISTORICAL-negative eval (Poursafaei et al., NeurIPS'22): negatives =
    #     u's TRAIN-period partners (seen historically) other than the true v.
    #     Tests whether the embedding ranks u's CURRENT/future partner above its
    #     PAST partners -- the eval where memorization/static struggles.
    train = edges[:cut]
    hist = {}
    for gu, gv in train[:, :2]:
        iu = idx_of.get(int(gu)); iv = idx_of.get(int(gv))
        if iu is None or iv is None:
            continue
        hist.setdefault(iu, set()).add(iv)
    hist = {u: np.fromiter(s, dtype=np.int64) for u, s in hist.items() if len(s) >= 2}

    # test positives whose source has >=2 historical partners
    cand = [(uu, vv) for uu, vv in zip(u_d, v_d) if uu in hist]
    if cand:
        cand = np.array(cand)
        if len(cand) > 20000:
            cand = cand[rng.choice(len(cand), 20000, replace=False)]
        Kn = 100
        negi = np.empty((len(cand), Kn), dtype=np.int64)
        for i, (uu, vv) in enumerate(cand):
            pool = hist[uu]
            negi[i] = pool[rng.integers(len(pool), size=Kn)]   # u's past partners
        uu = cand[:, 0]; vv = cand[:, 1]
        pos = np.sum(W[uu] * W[vv], axis=1)
        neg = np.einsum('pd,pkd->pk', W[uu], W[negi])
        ge = (neg >= pos[:, None]).sum(axis=1)
        mrr_h = float(np.mean(1.0 / (ge + 1)))
        h10_h = float(np.mean(ge < 10))
        print(f"HISTORICAL-neg LP: MRR={mrr_h:.4f}  Hits@10={h10_h:.4f}  "
              f"(negs = u's past train partners; {len(cand)} test edges)")


if __name__ == "__main__":
    main()
