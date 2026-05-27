#!/usr/bin/env python3
"""
partition_metis.py -- memory-efficient METIS partitioning for rw-temporal.

Reads a 3-column edge list `data/<basename>.txt` ("src dst t" per line) and
writes per-rank partition files into `data/<P>/`:

    data/<P>/<basename>.sub<r>.txt   -- local edges (src dst t)
    data/<P>/<basename>.rt<r>.txt    -- routing entries:
                                        `<src> "[(dst, proc, t), ...]"`

Usage:
    python3 partition_metis.py <edge_file> <num_parts>

Requirements:
    pip install pymetis numpy
    (pandas optional, used for ~5x faster file reading)

Memory notes:
  * Uses numpy int32 throughout to avoid Python-object overhead (28 B per
    int vs 4 B per int32 element). Tested at ~2 GB peak for 17M edges.
  * If you hit OOM on the full Stack-Overflow file (63M edges), consider
    using a smaller subset (a2q / c2q / c2a) or use a machine with >=8 GB
    available memory.
"""

import os
import sys
from collections import defaultdict

import numpy as np


def load_edges(path):
    """Return an (E, 3) int32 numpy array."""
    try:
        import pandas as pd
        df = pd.read_csv(path, sep=r'\s+', header=None, dtype=np.int32, engine='c')
        return df.values
    except ImportError:
        pass

    rows = []
    with open(path) as f:
        for line in f:
            parts = line.split()
            if len(parts) != 3:
                continue
            rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return np.array(rows, dtype=np.int32)


def partition_with_metis(xadj, adjncy, num_parts):
    """Call pymetis, accepting both the new (CSRAdjacency) and legacy APIs.
    Numpy arrays are passed directly (pymetis honours the buffer protocol)
    so we avoid building Python lists of tens of millions of ints. """
    import pymetis

    # New API (>=2024): pymetis.CSRAdjacency
    if hasattr(pymetis, "CSRAdjacency"):
        try:
            adj = pymetis.CSRAdjacency(xadj=xadj, adjncy=adjncy)
            return pymetis.part_graph(num_parts, adjacency=adj)
        except TypeError:
            pass  # fall through to legacy

    # Legacy API (still works, with DeprecationWarning)
    return pymetis.part_graph(num_parts, xadj=xadj, adjncy=adjncy)


def main():
    if len(sys.argv) != 3:
        sys.exit("usage: partition_metis.py <edge_file> <num_parts>")
    edge_file = sys.argv[1]
    num_parts = int(sys.argv[2])
    if not os.path.isfile(edge_file):
        sys.exit(f"file not found: {edge_file}")
    if num_parts < 2:
        sys.exit(f"num_parts must be >= 2 (got {num_parts})")

    # --- 1. Read edges
    print(f"[1/5] reading {edge_file}")
    edges = load_edges(edge_file)
    n_edges = len(edges)
    print(f"      edges={n_edges}")

    # --- 2. Densify node ids: flatten src/dst into one namespace
    print(f"[2/5] densifying node ids")
    flat = edges[:, :2].reshape(-1)
    unique, inv_flat = np.unique(flat, return_inverse=True)
    inv = inv_flat.astype(np.int32).reshape(-1, 2)
    del flat, inv_flat
    n_nodes = len(unique)
    print(f"      nodes={n_nodes}  avg_deg={2*n_edges/n_nodes:.1f}")

    # --- 3. CSR via sort-based scheme (vectorised, no per-edge Python loop)
    print(f"[3/5] building CSR")
    src = inv[:, 0]
    dst = inv[:, 1]
    all_src = np.concatenate([src, dst])
    all_dst = np.concatenate([dst, src])
    order = np.argsort(all_src, kind='stable')
    sorted_src = all_src[order]
    adjncy = all_dst[order].astype(np.int32)
    del all_src, all_dst, order

    degree = np.bincount(sorted_src, minlength=n_nodes).astype(np.int32)
    del sorted_src
    xadj = np.zeros(n_nodes + 1, dtype=np.int32)
    np.cumsum(degree, out=xadj[1:])
    del degree

    # --- 4. METIS
    print(f"[4/5] partitioning into {num_parts} parts")
    n_cuts, membership = partition_with_metis(xadj, adjncy, num_parts)
    del xadj, adjncy
    membership = np.asarray(membership, dtype=np.int32)
    print(f"      edge cuts = {n_cuts}  cut_ratio = {n_cuts / n_edges:.3f}")

    # --- 5. Bin edges into local-vs-cross, write per-rank files
    print(f"[5/5] writing per-rank files")
    base = os.path.basename(edge_file)
    if base.endswith(".txt"):
        base = base[:-4]
    out_dir = os.path.join(os.path.dirname(edge_file) or ".", str(num_parts))
    os.makedirs(out_dir, exist_ok=True)

    src_part = membership[inv[:, 0]]
    dst_part = membership[inv[:, 1]]
    is_local = (src_part == dst_part)
    n_cross = int(np.sum(~is_local))
    print(f"      cross-partition edges: {n_cross} ({100*n_cross/n_edges:.1f}%)")

    # 5a. Local edges per rank
    for r in range(num_parts):
        mask = is_local & (src_part == r)
        rank_edges = edges[mask]
        sub_path = os.path.join(out_dir, f"{base}.sub{r}.txt")
        np.savetxt(sub_path, rank_edges, fmt="%d %d %d")

    # 5b. Routing table per rank
    rt = [defaultdict(list) for _ in range(num_parts)]
    cross_idx = np.where(~is_local)[0]
    src_orig = edges[:, 0]
    dst_orig = edges[:, 1]
    t_arr    = edges[:, 2]

    # Python loop over cross edges only (much smaller than n_edges).
    for i in cross_idx.tolist():
        u = int(src_orig[i]); v = int(dst_orig[i]); t = int(t_arr[i])
        ru = int(src_part[i]); rv = int(dst_part[i])
        rt[ru][u].append((v, rv, t))
        rt[rv][v].append((u, ru, t))

    for r in range(num_parts):
        rt_path = os.path.join(out_dir, f"{base}.rt{r}.txt")
        with open(rt_path, "w") as f:
            for u, peers in rt[r].items():
                peers_str = ", ".join(f"({v}, {ru}, {t})" for v, ru, t in peers)
                f.write(f'{u} "[{peers_str}]"\n')

        owned = int(np.sum(membership == r))
        n_local_edges = int(np.sum(is_local & (src_part == r)))
        print(f"      rank {r}: owned={owned}  local_edges={n_local_edges}  "
              f"boundary_nodes={len(rt[r])}")

    print(f"\ndone.  output in {out_dir}/")


if __name__ == "__main__":
    main()
