#!/usr/bin/env python3
"""
analyze_scaling.py -- parse Wisteria scaling job outputs (rwscale*.out) into a
median[min-max] table across repetitions, and a static-vs-T4 (or any two
datasets) side-by-side comparison.

Each .out file is ONE run (one scaling point, one rep). Runs are grouped by
(dataset, proc); across the reps in each group we report median and [min,max].

Usage:
  analyze_scaling.py <glob> [<glob> ...]
  e.g. analyze_scaling.py 't4weak.*.out' 'rwscale.90519*.out' 'rwscale.9052009.out'
"""
import glob, re, sys, statistics as st

RE_HEAD = re.compile(r"=== scaling:.*?proc=(\d+)\s+dataset=(\S+)")
RE_PHASE = re.compile(r"PHASE compute=([\d.]+) exchange=([\d.]+) allreduce=([\d.]+) "
                      r"emb_xchg=([\d.]+) s\s+comm_frac=([\d.]+)%.*mode=(\w+)")
RE_E1 = re.compile(r"E1 cross-rank pairs: \d+/\d+ \(([\d.]+)%\)")
RE_EL = re.compile(r"elapsed=([\d.]+)s")


def parse(path):
    proc = ds = mode = None
    rec = {}
    with open(path, errors="ignore") as f:
        for line in f:
            m = RE_HEAD.search(line)
            if m: proc, ds = int(m.group(1)), m.group(2)
            m = RE_PHASE.search(line)
            if m:
                rec["compute"], rec["exchange"], rec["allreduce"], rec["emb_xchg"], \
                    rec["comm"] = map(float, m.group(1, 2, 3, 4, 5))
                mode = m.group(6)
            m = RE_E1.search(line)
            if m: rec["cross"] = float(m.group(1))
            m = RE_EL.search(line)
            if m: rec["elapsed"] = float(m.group(1))
    if proc is None or "elapsed" not in rec:
        return None
    rec.update(proc=proc, dataset=ds, mode=mode)
    return rec


def fmt(vals):
    if not vals: return "-"
    md = st.median(vals)
    return f"{md:.2f}[{min(vals):.2f}-{max(vals):.2f}]" if len(vals) > 1 else f"{md:.2f}"


def main():
    files = []
    for g in sys.argv[1:]:
        files += glob.glob(g)
    recs = [r for r in (parse(p) for p in sorted(set(files))) if r]
    if not recs:
        sys.exit("no parseable runs (need PHASE + elapsed lines)")

    # group by VARIANT = (dataset, mode) and proc, so fused vs twostage on the
    # same dataset are separate groups (and static vs T4 are too).
    groups = {}
    for r in recs:
        groups.setdefault((r["dataset"], r["mode"], r["proc"]), []).append(r)

    cols = ["elapsed", "compute", "exchange", "allreduce", "emb_xchg", "comm", "cross"]
    variants = sorted({(d, m) for d, m, _ in groups})
    for (ds, mode) in variants:
        print(f"\n=== {ds} / {mode} ===")
        print(f"{'proc':>5} {'reps':>4}  " + "  ".join(f"{c:>16}" for c in cols))
        for (d, m, p) in sorted(groups):
            if (d, m) != (ds, mode): continue
            g = groups[(d, m, p)]
            row = [fmt([x[c] for x in g if c in x]) for c in cols]
            print(f"{p:>5} {len(g):>4}  " + "  ".join(f"{v:>16}" for v in row))

    # pairwise comparison if exactly two variants (static vs T4, or fused vs twostage)
    if len(variants) == 2:
        (da, ma), (db, mb) = variants
        la, lb = f"{da}/{ma}", f"{db}/{mb}"
        print(f"\n=== {lb}  vs  {la}  (median delta) ===")
        print(f"{'proc':>5}  {'cross Δpt':>11}  {'exch %':>9}  {'elapsed %':>10}  "
              f"{'emb_xchg a->b (s)':>20}")
        procs = sorted({p for (d, m, p) in groups if (d, m) == (da, ma)}
                       & {p for (d, m, p) in groups if (d, m) == (db, mb)})
        for p in procs:
            ga, gb = groups[(da, ma, p)], groups[(db, mb, p)]
            def med(g, c): return st.median([x[c] for x in g if c in x])
            dcross = med(gb, "cross") - med(ga, "cross")
            dexch = 100 * (med(gb, "exchange") / med(ga, "exchange") - 1)
            dela = 100 * (med(gb, "elapsed") / med(ga, "elapsed") - 1)
            ea, eb = med(ga, "emb_xchg"), med(gb, "emb_xchg")
            print(f"{p:>5}  {dcross:>+10.1f}  {dexch:>+8.1f}  {dela:>+9.1f}  "
                  f"{ea:>8.3f} -> {eb:<8.3f}")


if __name__ == "__main__":
    main()
