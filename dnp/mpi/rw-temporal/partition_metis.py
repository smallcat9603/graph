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

# Keep in sync with config.h MAX_CHUNK_BYTES (90 MiB).
MAX_CHUNK_BYTES = 90 * 1024 * 1024


def resolve_chunks(logical):
    """Resolve a logical path to its on-disk chunk file(s).
    Single file if it exists, else contiguous <logical>.part000, .part001, ..."""
    if os.path.isfile(logical):
        return [logical]
    parts = []
    i = 0
    while True:
        p = f"{logical}.part{i:03d}"
        if not os.path.isfile(p):
            break
        parts.append(p)
        i += 1
    return parts


def write_split(path, line_iter):
    """Write text lines (each already ending in '\\n') to <path>, splitting at
    line boundaries into <path>.part000, .part001, ... whenever a part would
    exceed MAX_CHUNK_BYTES. If only one part results, rename it to <path>."""
    # Clear any stale output from a previous run (single file + parts).
    if os.path.exists(path):
        os.remove(path)
    i = 0
    while True:
        sp = f"{path}.part{i:03d}"
        if not os.path.exists(sp):
            break
        os.remove(sp)
        i += 1
    part_idx = 0
    part_bytes = 0
    fp = open(f"{path}.part{part_idx:03d}", "w")
    for line in line_iter:
        b = len(line.encode("utf-8"))
        if part_bytes + b > MAX_CHUNK_BYTES and part_bytes > 0:
            fp.close()
            part_idx += 1
            part_bytes = 0
            fp = open(f"{path}.part{part_idx:03d}", "w")
        fp.write(line)
        part_bytes += b
    fp.close()
    if part_idx == 0:
        os.replace(f"{path}.part000", path)


def load_edges(path):
    """Return an (E, 3) int32 numpy array, reading across chunks if split."""
    chunks = resolve_chunks(path)
    if not chunks:
        sys.exit(f"file not found: {path} (or {path}.part000)")
    try:
        import pandas as pd
        frames = [pd.read_csv(c, sep=r'\s+', header=None, dtype=np.int32, engine='c')
                  for c in chunks]
        return np.concatenate([f.values for f in frames], axis=0)
    except ImportError:
        pass

    rows = []
    for c in chunks:
        with open(c) as f:
            for line in f:
                parts = line.split()
                if len(parts) != 3:
                    continue
                rows.append((int(parts[0]), int(parts[1]), int(parts[2])))
    return np.array(rows, dtype=np.int32)


def partition_with_metis(xadj, adjncy, num_parts, eweights=None):
    """Call pymetis, accepting both the new (CSRAdjacency) and legacy APIs.
    Numpy arrays are passed directly (pymetis honours the buffer protocol)
    so we avoid building Python lists of tens of millions of ints.

    When eweights is given (adjwgt aligned with adjncy), METIS minimizes the
    *weighted* edgecut, so high-weight edges are kept within a partition. We
    use the legacy call for the weighted case (it reliably accepts eweights)."""
    import pymetis

    if eweights is not None:
        return pymetis.part_graph(num_parts, xadj=xadj, adjncy=adjncy,
                                  eweights=eweights)

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
    if len(sys.argv) not in (3, 4):
        sys.exit("usage: partition_metis.py <edge_file> <num_parts> "
                 "[temporal_scale]\n"
                 "  temporal_scale > 0 (e.g. 100): T4 prototype -- weight edges\n"
                 "  by earliness in time (early edges, which time-respecting\n"
                 "  walks traverse most, are kept within a partition). Output\n"
                 "  goes to <base>_tw.* so the unweighted baseline is preserved.")
    edge_file = sys.argv[1]
    num_parts = int(sys.argv[2])
    temporal_scale = float(sys.argv[3]) if len(sys.argv) == 4 else 0.0
    if not resolve_chunks(edge_file):
        sys.exit(f"file not found: {edge_file} (or {edge_file}.part000)")
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
    del all_src, all_dst   # keep `order` for temporal-weight alignment below

    degree = np.bincount(sorted_src, minlength=n_nodes).astype(np.int32)
    del sorted_src
    xadj = np.zeros(n_nodes + 1, dtype=np.int32)
    np.cumsum(degree, out=xadj[1:])
    del degree

    # --- 3b. T4 temporal edge weights (prototype). Time-respecting walks
    #         dead-end fast (cursor races to t_max), so they overwhelmingly
    #         traverse EARLY-timestamp edges. Weight early edges high so METIS
    #         keeps the actually-traversed edges within a partition.
    adjwgt = None
    eweight_file = os.environ.get("EWEIGHT_FILE")
    if eweight_file:
        # Empirical per-edge weights (pilot_edge_weights.py): one weight per
        # edge in the edge-file row order. Align with adjncy via the same
        # doubling + ordering used for the CSR.
        edge_w = np.loadtxt(eweight_file, dtype=np.int32)
        if len(edge_w) != n_edges:
            sys.exit(f"EWEIGHT_FILE has {len(edge_w)} rows, expected {n_edges}")
        all_w = np.concatenate([edge_w, edge_w])
        adjwgt = all_w[order].astype(np.int32)
        del all_w, edge_w
        temporal_scale = 1.0  # mark weighted so output goes to <base>_tw
        print(f"      empirical edge weights from {eweight_file} "
              f"(w in [{int(adjwgt.min())}, {int(adjwgt.max())}])")
    elif temporal_scale > 0:
        t = edges[:, 2].astype(np.float64)
        tmin, tmax = float(t.min()), float(t.max())
        span = (tmax - tmin) if tmax > tmin else 1.0
        t_norm = (t - tmin) / span                      # 0=earliest, 1=latest
        edge_w = (1.0 + temporal_scale * (1.0 - t_norm)) # early -> high weight
        edge_w = np.rint(edge_w).astype(np.int32)
        np.maximum(edge_w, 1, out=edge_w)
        # Align with adjncy: same doubling (src->dst, dst->src) then same order.
        all_w = np.concatenate([edge_w, edge_w])
        adjwgt = all_w[order].astype(np.int32)
        del all_w, t, t_norm, edge_w
        print(f"      temporal weights: scale={temporal_scale} "
              f"w in [1, {int(1+temporal_scale)}], early edges weighted high")
    del order

    # --- 4. METIS
    print(f"[4/5] partitioning into {num_parts} parts"
          f"{' (temporal-weighted)' if adjwgt is not None else ''}")
    n_cuts, membership = partition_with_metis(xadj, adjncy, num_parts, adjwgt)
    del xadj, adjncy, adjwgt
    membership = np.asarray(membership, dtype=np.int32)
    print(f"      edge cuts = {n_cuts}  cut_ratio = {n_cuts / n_edges:.3f}")

    # --- 5. Bin edges into local-vs-cross, write per-rank files
    print(f"[5/5] writing per-rank files")
    base = os.path.basename(edge_file)
    if base.endswith(".txt"):
        base = base[:-4]
    if temporal_scale > 0:
        base = base + "_tw"   # keep the unweighted baseline partition intact
    out_dir = os.path.join(os.path.dirname(edge_file) or ".", str(num_parts))
    os.makedirs(out_dir, exist_ok=True)

    src_part = membership[inv[:, 0]]
    dst_part = membership[inv[:, 1]]
    is_local = (src_part == dst_part)
    n_cross = int(np.sum(~is_local))
    print(f"      cross-partition edges: {n_cross} ({100*n_cross/n_edges:.1f}%)")

    # 5a. Local edges per rank (split into <100MB chunks)
    for r in range(num_parts):
        mask = is_local & (src_part == r)
        rank_edges = edges[mask]
        sub_path = os.path.join(out_dir, f"{base}.sub{r}.txt")
        write_split(sub_path,
                    (f"{e[0]} {e[1]} {e[2]}\n" for e in rank_edges))

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

    def rt_lines(entries):
        for u, peers in entries.items():
            peers_str = ", ".join(f"({v}, {ru}, {t})" for v, ru, t in peers)
            yield f'{u} "[{peers_str}]"\n'

    for r in range(num_parts):
        rt_path = os.path.join(out_dir, f"{base}.rt{r}.txt")
        write_split(rt_path, rt_lines(rt[r]))

        owned = int(np.sum(membership == r))
        n_local_edges = int(np.sum(is_local & (src_part == r)))
        print(f"      rank {r}: owned={owned}  local_edges={n_local_edges}  "
              f"boundary_nodes={len(rt[r])}")

    print(f"\ndone.  output in {out_dir}/")


if __name__ == "__main__":
    main()
