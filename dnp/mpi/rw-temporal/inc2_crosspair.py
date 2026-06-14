#!/usr/bin/env python3
"""
inc2_crosspair.py -- quantify the VALUE of inc-2 (training cross-partition
positive pairs) before building the distributed migration-piggyback.

Single-machine, but uses the REAL METIS partition (not random shards), so the
cross-pair fraction matches what the engine sees. Trains SGNS two ways on the
same walks + same METIS partition, with global negatives (to isolate the
positive-pair effect from the negative-sampling effect):
    keep : all window pairs                       (= inc-2, cross pairs trained)
    drop : same-shard pairs only                  (= inc-1, cross pairs dropped)
Reports temporal-LP AP for each + the cross-pair fraction. A small keep-vs-drop
gap means local-only (inc-1) suffices and the expensive piggyback is not worth
building for that graph; a large gap justifies inc-2.

Usage: inc2_crosspair.py <edge_file> [n_shards] [dim] [epochs]
"""
import sys
import numpy as np
sys.path.insert(0, __file__.rsplit('/', 1)[0] if '/' in __file__ else '.')
from partition_metis import load_edges, partition_with_metis
from e2_negsample import (build_walks, unigram_table, train_sgns, eval_lp,
                          detect_dst_pool)

RNG = np.random.default_rng(0)


def metis_membership(inv, n, n_parts):
    src, dst = inv[:, 0], inv[:, 1]
    all_src = np.concatenate([src, dst]); all_dst = np.concatenate([dst, src])
    order = np.argsort(all_src, kind='stable')
    adjncy = all_dst[order].astype(np.int32)
    deg = np.bincount(all_src[order], minlength=n).astype(np.int32)
    xadj = np.zeros(n + 1, dtype=np.int32); np.cumsum(deg, out=xadj[1:])
    _, membership = partition_with_metis(xadj, adjncy, n_parts)
    return np.asarray(membership, dtype=np.int32)


def main():
    edge_file = sys.argv[1]
    n_shards  = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    dim       = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    epochs    = int(sys.argv[4]) if len(sys.argv) > 4 else 3

    edges = load_edges(edge_file)
    edges = edges[np.argsort(edges[:, 2], kind='stable')]
    nodes, inv = np.unique(edges[:, :2].reshape(-1), return_inverse=True)
    inv = inv.reshape(-1, 2); n = len(nodes)
    split = int(0.8 * len(edges))
    tr, te = inv[:split], inv[split:]
    tr_t = edges[:split, 2]

    membership = metis_membership(tr, n, n_shards)

    adj = [[] for _ in range(n)]
    for i in range(len(tr)):
        a, b, t_ = int(tr[i, 0]), int(tr[i, 1]), int(tr_t[i])
        adj[a].append((t_, b)); adj[b].append((t_, a))
    for a in range(n):
        adj[a].sort()
    adj_t = [np.array([e[0] for e in adj[a]], dtype=np.int64) for a in range(n)]

    # Cap total walks so huge graphs don't explode the positive-pair list
    # (n*5 walks on 2.46M nodes -> ~240M pairs -> tens of GB). Small graphs
    # (n*5 <= cap) are unaffected, so wikipedia/reddit stay comparable.
    MAX_WALKS = 400_000
    if n * 5 <= MAX_WALKS:
        walks = build_walks(adj, adj_t, n, n_walks=5, max_steps=10)
    else:
        starts = RNG.integers(n, size=MAX_WALKS)
        walks = []
        for s in starts:
            cur, tcur, path = int(s), -1, [int(s)]
            for _ in range(10):
                times = adj_t[cur]
                lo = int(np.searchsorted(times, tcur, side='right'))
                ncand = len(times) - lo
                if ncand <= 0:
                    break
                te_, nb = adj[cur][lo + int(RNG.integers(ncand))]
                path.append(nb); tcur = te_; cur = nb
            if len(path) >= 2:
                walks.append(path)
        print(f"(capped to {MAX_WALKS} walks for large graph)")
    W_WIN = 5
    pp = []
    for w in walks:
        L = len(w)
        for i in range(L):
            for j in range(max(0, i - W_WIN), min(L, i + W_WIN + 1)):
                if i != j:
                    pp.append((w[i], w[j]))
    pairs = np.array(pp, dtype=np.int64)
    same = membership[pairs[:, 0]] == membership[pairs[:, 1]]
    cross_frac = 100.0 * (1 - same.mean())
    print(f"{edge_file}: nodes={n} pairs={len(pairs)} shards={n_shards} "
          f"cross-pair={cross_frac:.1f}%")

    freq = np.bincount(pairs[:, 0], minlength=n) + 1
    gtab = unigram_table(freq, np.arange(n))
    d = dim
    W0 = (RNG.random((n, d)) - 0.5) / d
    C0 = np.zeros((n, d))
    dummy_sh = np.zeros(n, dtype=np.int64); dummy_M = np.ones(1)
    rng_eval = np.random.default_rng(7)
    dst_pool = detect_dst_pool(inv, n)

    print(f"\n{'pairs':<8}{'AUC':>8}{'MRR':>8}{'H@10':>8}")
    for label, pset in [("keep", pairs), ("drop", pairs[same])]:
        W = train_sgns(pset, W0, C0, dummy_sh, [gtab], dummy_M, "global",
                       epochs=epochs, global_table=gtab)
        auc, mrr, h10 = eval_lp(W, te, n, rng_eval, dst_pool=dst_pool)
        print(f"{label:<8}{auc:>8.4f}{mrr:>8.4f}{h10:>8.4f}   ({len(pset)} pairs)")


if __name__ == "__main__":
    main()
