#!/usr/bin/env python3
"""
pilot_edge_weights.py -- T4 prototype: empirical time-respecting traversal
frequency as a per-edge weight for partitioning.

Loads a 3-column edge list (src dst t), simulates K forward time-respecting
walks on the FULL graph (no partition needed), and counts how many times each
edge is traversed. Writes one integer weight per edge (1 + traversal_count),
in the SAME row order as the edge file, to <edge_file>.ew.

This is the principled "expected time-respecting traversal frequency" the plan
specifies (the cheap analytic 'earliness' weight failed E1). partition_metis.py
reads this file via the EWEIGHT_FILE env var.

Usage: pilot_edge_weights.py <edge_file> [n_walks] [max_steps]
"""
import os
import sys
import numpy as np

# reuse the chunk-aware loader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from partition_metis import load_edges


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: pilot_edge_weights.py <edge_file> [n_walks] [max_steps]")
    edge_file = sys.argv[1]
    n_walks   = int(sys.argv[2]) if len(sys.argv) > 2 else 20000
    max_steps = int(sys.argv[3]) if len(sys.argv) > 3 else 30

    edges = load_edges(edge_file)          # (E,3): src dst t
    E = len(edges)
    src, dst, t = edges[:, 0], edges[:, 1], edges[:, 2]

    # Densify node ids.
    nodes, inv = np.unique(edges[:, :2].reshape(-1), return_inverse=True)
    inv = inv.reshape(-1, 2)
    n = len(nodes)
    u_local, v_local = inv[:, 0], inv[:, 1]

    # Build per-node time-sorted adjacency: lists of (t, neighbor_local, edge_id),
    # each undirected edge appearing for both endpoints.
    adj = [[] for _ in range(n)]
    for eid in range(E):
        a, b, te = int(u_local[eid]), int(v_local[eid]), int(t[eid])
        adj[a].append((te, b, eid))
        adj[b].append((te, a, eid))
    for a in range(n):
        adj[a].sort()                      # by time
    # Precompute sorted time arrays for binary search.
    adj_t = [np.array([e[0] for e in adj[a]], dtype=np.int64) for a in range(n)]

    rng = np.random.default_rng(0)
    count = np.zeros(E, dtype=np.int64)
    lengths = np.zeros(n_walks, dtype=np.int32)  # hops actually taken per walk
    for wkr in range(n_walks):
        cur = int(rng.integers(n))
        tcur = -1
        steps = 0
        for _step in range(max_steps):
            times = adj_t[cur]
            lo = int(np.searchsorted(times, tcur, side='right'))  # upper_bound
            ncand = len(times) - lo
            if ncand <= 0:
                break
            pick = lo + int(rng.integers(ncand))
            te, nb, eid = adj[cur][pick]
            count[eid] += 1
            tcur = te
            cur = nb
            steps += 1
        lengths[wkr] = steps

    weight = (1 + count).astype(np.int64)
    out = edge_file + ".ew"
    np.savetxt(out, weight, fmt="%d")
    traversed = int(np.sum(count > 0))
    # Walk-length distribution: is the walk non-trivial, or mostly length-1?
    mean_len = float(lengths.mean())
    med_len  = int(np.median(lengths))
    ge = {k: 100.0 * float(np.mean(lengths >= k)) for k in (1, 2, 5, 10)}
    print(f"walks={n_walks} steps<= {max_steps}  edges={E}  "
          f"traversed={traversed} ({100*traversed/E:.1f}%)  max_count={int(count.max())}")
    print(f"walk length: mean={mean_len:.2f} median={med_len}  "
          f">=1:{ge[1]:.0f}%  >=2:{ge[2]:.0f}%  >=5:{ge[5]:.0f}%  >=10:{ge[10]:.0f}%")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
