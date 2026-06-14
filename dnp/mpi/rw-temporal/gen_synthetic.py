#!/usr/bin/env python3
"""
gen_synthetic.py -- stream a large scale-free TEMPORAL edge list to disk for the
OOM-capability demo (a graph too big to hold on one 28 GiB node, but fine when
partitioned across many).

Memory-light: nodes are sampled by inverse-CDF of a power-law degree
distribution (one precomputed cumulative array), edges are generated and written
in fixed-size chunks, and the output is split into <out>.partNNN at ~90 MiB
boundaries (the repo's chunk convention; load_edges/resolve_chunks read it back
transparently).

Each line is "src dst t" (t in [0, t_max)).

Usage:
  gen_synthetic.py <out.txt> <n_nodes> <n_edges> [alpha] [t_max] [seed]
  e.g. (≈1.2B-edge, ~14 GB):  gen_synthetic.py /work/g/synth.txt 50000000 1200000000
"""
import os, sys
import numpy as np

MAX_CHUNK_BYTES = 90 * 1024 * 1024


def main():
    if len(sys.argv) < 4:
        sys.exit("usage: gen_synthetic.py <out.txt> <n_nodes> <n_edges> "
                 "[alpha=0.8] [t_max=2_000_000_000] [seed=0]")
    out      = sys.argv[1]
    n_nodes  = int(sys.argv[2])
    n_edges  = int(sys.argv[3])
    alpha    = float(sys.argv[4]) if len(sys.argv) > 4 else 0.8
    t_max    = int(sys.argv[5])   if len(sys.argv) > 5 else 2_000_000_000
    seed     = int(sys.argv[6])   if len(sys.argv) > 6 else 0
    rng = np.random.default_rng(seed)

    # power-law node weights w_i ∝ (i+1)^(-alpha); inverse-CDF sampling
    ranks = np.arange(1, n_nodes + 1, dtype=np.float64)
    w = ranks ** (-alpha)
    cdf = np.cumsum(w); cdf /= cdf[-1]

    def sample(k):
        return np.searchsorted(cdf, rng.random(k)).astype(np.int64)

    # clear any stale output (single file + chunks)
    for p in [out] + [f"{out}.part{i:03d}" for i in range(10000)]:
        if os.path.exists(p):
            os.remove(p)
        elif p != out and not os.path.exists(p):
            if p.endswith("part0000"):  # nothing to clear past first gap
                pass

    CHUNK = 5_000_000
    part_idx, part_bytes = 0, 0
    fp = open(f"{out}.part{part_idx:03d}", "w")
    written = 0
    while written < n_edges:
        k = min(CHUNK, n_edges - written)
        src = sample(k); dst = sample(k)
        # avoid self-loops
        m = src == dst
        if m.any():
            dst[m] = (dst[m] + 1) % n_nodes
        t = rng.integers(0, t_max, size=k, dtype=np.int64)
        # format block as text
        block = "\n".join(f"{int(a)} {int(b)} {int(c)}"
                          for a, b, c in zip(src, dst, t)) + "\n"
        b = len(block.encode())
        if part_bytes + b > MAX_CHUNK_BYTES and part_bytes > 0:
            fp.close(); part_idx += 1; part_bytes = 0
            fp = open(f"{out}.part{part_idx:03d}", "w")
        fp.write(block); part_bytes += b; written += k
        if written % 50_000_000 < CHUNK:
            print(f"  {written:,}/{n_edges:,} edges ({part_idx+1} chunks)")
    fp.close()
    # single tiny output -> rename to logical name
    if part_idx == 0:
        os.replace(f"{out}.part000", out)
    print(f"done: {n_edges:,} edges, {part_idx+1} chunk(s), nodes={n_nodes:,}")


if __name__ == "__main__":
    main()
