#!/usr/bin/env python3
"""
e2_negsample.py -- E2 de-risking (research_plan_v3.md, Innovation 2 / T3).

Single-machine ML check, NO MPI. Does shard-local negative sampling keep
embedding quality, once corrected by the importance weight?

Pipeline: chronological train/test split -> time-respecting walks on TRAIN ->
skip-gram (SGNS) embeddings under 3-4 negative-sampling regimes -> temporal
link-prediction AUC / AP. Everything except the negatives is held identical
(same walks, same positive pairs, same init, same RNG) so the only variable is
how negatives are drawn.

Regimes:
  global   : negatives ~ global unigram^0.75            (reference upper bound)
  local    : negatives ~ local shard only, NO correction (naive distributed)
  local_iw : local shard + importance weight  w = M_s   (the proposed fix)
  local_iw_x: local + weight + periodic cross-shard exchange (rate rho)

Usage: e2_negsample.py <edge_file> [n_shards] [dim] [epochs]
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from partition_metis import load_edges

RNG = np.random.default_rng(0)


def build_walks(adj, adj_t, n, n_walks, max_steps):
    walks = []
    for s in range(n):
        for _ in range(n_walks):
            cur, tcur, path = s, -1, [s]
            for _ in range(max_steps):
                times = adj_t[cur]
                lo = int(np.searchsorted(times, tcur, side='right'))
                ncand = len(times) - lo
                if ncand <= 0:
                    break
                te, nb = adj[cur][lo + int(RNG.integers(ncand))]
                path.append(nb); tcur = te; cur = nb
            if len(path) >= 2:
                walks.append(path)
    return walks


def unigram_table(freq, node_ids, power=0.75, size=10_000_000):
    p = freq.astype(np.float64) ** power
    p /= p.sum()
    # expected-count table: repeat each node id ~ size*p[i] times
    counts = np.maximum(1, np.round(p * size)).astype(np.int64)
    return np.repeat(node_ids, counts)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def train_sgns(pairs, W0, C0, shard_of, tables, M, regime, K=5,
               epochs=3, lr0=0.025, rho=0.1, global_table=None, seed=42):
    """pairs: (P,2) int (center, context). Returns trained input embeddings."""
    W = W0.copy(); C = C0.copy()
    P = len(pairs)
    rng = np.random.default_rng(seed)
    bs = 4096
    for ep in range(epochs):
        lr = lr0 * (1 - ep / (epochs + 1))
        order = rng.permutation(P)
        for b0 in range(0, P, bs):
            idx = order[b0:b0 + bs]
            u = pairs[idx, 0]; v = pairs[idx, 1]
            B = len(u)
            # --- negatives + per-negative weight
            neg = np.empty((B, K), dtype=np.int64)
            wgt = np.ones((B, K), dtype=np.float64)
            if regime == "global":
                neg = global_table[rng.integers(len(global_table), size=(B, K))]
            else:
                cs = shard_of[u]
                for s in np.unique(cs):
                    m = np.where(cs == s)[0]
                    tab = tables[s]
                    draw = tab[rng.integers(len(tab), size=(len(m), K))]
                    if regime.endswith("_x"):  # periodic cross-shard exchange
                        ex = rng.random((len(m), K)) < rho
                        gd = global_table[rng.integers(len(global_table),
                                                       size=(len(m), K))]
                        draw = np.where(ex, gd, draw)
                        neg[m] = draw
                        wgt[m] = np.where(ex, 1.0, M[s])
                    else:
                        neg[m] = draw
                        if regime == "local_iw":
                            wgt[m] = M[s]
                        # regime == "local": weight stays 1 (no correction)
            # --- SGNS gradients (vectorized)
            zu = W[u]                      # (B,d)
            zv = C[v]                      # (B,d)
            pos = sigmoid(np.sum(zu * zv, axis=1))         # (B,)
            g_pos = (pos - 1.0)[:, None]                    # (B,1)
            zn = C[neg]                    # (B,K,d)
            sn = sigmoid(np.einsum('bd,bkd->bk', zu, zn))  # (B,K)
            g_neg = (sn * wgt)[:, :, None]                  # (B,K,1) weighted
            # grad wrt u: g_pos*zv + sum_k g_neg*zn
            gu = g_pos * zv + np.einsum('bk,bkd->bd', g_neg[:, :, 0], zn)
            gv = g_pos * zu
            gn = g_neg * zu[:, None, :]                     # (B,K,d)
            # apply
            W[u] -= lr * gu
            np.add.at(C, v, -lr * gv)
            np.add.at(C, neg.reshape(-1), (-lr * gn).reshape(-1, gn.shape[2]))
    return W


def eval_lp(W, test_pos, n, rng, dst_pool=None, Kn=100):
    """Destination-ranking temporal LP (TGB/CTDNE style). For each positive
    (u, v), draw Kn negative destinations from dst_pool (the correct node type
    for bipartite graphs, else all nodes) and rank v against them by dot score.
    Returns (per-edge AUC, MRR, Hits@10). Batched to bound memory."""
    u = test_pos[:, 0]; v = test_pos[:, 1]
    if dst_pool is None:
        dst_pool = np.arange(W.shape[0])
    P = len(u)
    inv_rank = np.empty(P); hits = np.empty(P); aucs = np.empty(P)
    B = 4096
    for b0 in range(0, P, B):
        ub = u[b0:b0 + B]; vb = v[b0:b0 + B]; m = len(ub)
        pos = np.sum(W[ub] * W[vb], axis=1)                       # (m,)
        negi = dst_pool[rng.integers(len(dst_pool), size=(m, Kn))]
        neg = np.einsum('pd,pkd->pk', W[ub], W[negi])             # (m,Kn)
        ge = (neg >= pos[:, None]).sum(axis=1)                    # negs beating pos
        rank = ge + 1
        inv_rank[b0:b0 + m] = 1.0 / rank
        hits[b0:b0 + m] = (rank <= 10)
        aucs[b0:b0 + m] = (neg < pos[:, None]).mean(axis=1)
    return float(aucs.mean()), float(inv_rank.mean()), float(hits.mean())


def detect_dst_pool(inv, n):
    """If the graph is (near-)bipartite, return the set of destination-side
    local node ids as the negative pool; else None (use all nodes)."""
    src = set(np.unique(inv[:, 0]).tolist())
    dst = set(np.unique(inv[:, 1]).tolist())
    overlap = len(src & dst)
    if overlap <= 0.02 * min(len(src), len(dst)):       # < 2% overlap => bipartite
        return np.array(sorted(dst), dtype=np.int64)
    return None


def main():
    global RNG
    edge_file = sys.argv[1] if len(sys.argv) > 1 else "data/wikipedia.txt"
    n_shards  = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    dim       = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    epochs    = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    import os as _os0
    seed = int(_os0.environ.get("E2_SEED", "0"))
    RNG = np.random.default_rng(seed)

    edges = load_edges(edge_file)
    edges = edges[np.argsort(edges[:, 2], kind='stable')]   # chronological
    nodes, inv = np.unique(edges[:, :2].reshape(-1), return_inverse=True)
    inv = inv.reshape(-1, 2); n = len(nodes)
    t = edges[:, 2]

    split = int(0.8 * len(edges))
    tr, te = inv[:split], inv[split:]
    tr_t = t[:split]
    print(f"{os.path.basename(edge_file)}: nodes={n} edges={len(edges)} "
          f"train={len(tr)} test={len(te)} shards={n_shards} dim={dim} ep={epochs}")

    # train adjacency (time-sorted, undirected)
    adj = [[] for _ in range(n)]
    for i in range(len(tr)):
        a, b, te_ = int(tr[i, 0]), int(tr[i, 1]), int(tr_t[i])
        adj[a].append((te_, b)); adj[b].append((te_, a))
    for a in range(n):
        adj[a].sort()
    adj_t = [np.array([e[0] for e in adj[a]], dtype=np.int64) for a in range(n)]

    walks = build_walks(adj, adj_t, n, n_walks=5, max_steps=10)
    # positive pairs from walks, window=5
    W_WIN = 5
    pp = []
    for w in walks:
        L = len(w)
        for i in range(L):
            for j in range(max(0, i - W_WIN), min(L, i + W_WIN + 1)):
                if i != j:
                    pp.append((w[i], w[j]))
    pairs = np.array(pp, dtype=np.int64)
    print(f"walks={len(walks)} positive_pairs={len(pairs)}")

    # frequencies (node occurrence in walks), shards, tables, masses
    freq = np.bincount(pairs[:, 0], minlength=n) + 1
    import os as _os
    if _os.environ.get("E2_SHARDS", "metis") == "random":
        shard_of = RNG.integers(n_shards, size=n)      # balanced random shards
        print("shards: random (balanced)")
    else:
        from partition_metis import partition_with_metis
        a_s = np.concatenate([inv[:, 0], inv[:, 1]])
        a_d = np.concatenate([inv[:, 1], inv[:, 0]])
        o = np.argsort(a_s, kind='stable')
        adjncy = a_d[o].astype(np.int32)
        deg = np.bincount(a_s[o], minlength=n).astype(np.int32)
        xadj = np.zeros(n + 1, dtype=np.int32); np.cumsum(deg, out=xadj[1:])
        _, mem = partition_with_metis(xadj, adjncy, n_shards)
        shard_of = np.asarray(mem, dtype=np.int64)
        print(f"shards: METIS (sizes {np.bincount(shard_of, minlength=n_shards)})")
    node_ids = np.arange(n)
    global_table = unigram_table(freq, node_ids)
    p_glob = (freq.astype(np.float64) ** 0.75); p_glob /= p_glob.sum()
    tables = []; M = np.zeros(n_shards)
    for s in range(n_shards):
        sm = np.where(shard_of == s)[0]
        tables.append(unigram_table(freq[sm], sm))
        M[s] = p_glob[sm].sum()                        # shard global mass
    print(f"shard masses M = {np.round(M, 3)}  (sum={M.sum():.3f})")

    d = dim
    W0 = (RNG.random((n, d)) - 0.5) / d
    C0 = np.zeros((n, d))
    rng_eval = np.random.default_rng(7)
    dst_pool = detect_dst_pool(inv, n)
    print(f"eval: {'bipartite (type-aware negs, |dst|=%d)' % len(dst_pool) if dst_pool is not None else 'general (all-node negs)'}")

    import os as _os
    rho = float(_os.environ.get("E2_RHO", "0.1"))
    regimes = _os.environ.get("E2_REGIMES", "global,local,local_iw,local_iw_x").split(",")
    print(f"\n{'regime':<14}{'AUC':>8}{'MRR':>8}{'H@10':>8}   (dest-ranking LP; rho={rho})")
    for regime in regimes:
        t0 = time.time()
        W = train_sgns(pairs, W0, C0, shard_of, tables, M, regime,
                       epochs=epochs, rho=rho, global_table=global_table, seed=seed + 42)
        auc, mrr, h10 = eval_lp(W, te, n, rng_eval, dst_pool=dst_pool)
        print(f"{regime:<14}{auc:>8.4f}{mrr:>8.4f}{h10:>8.4f}   "
              f"rho={rho} ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
