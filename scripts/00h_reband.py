#!/usr/bin/env python3
"""00h_reband.py -- ZERO API calls.

The current 'headroom' set was defined as 0 < pass_rate < 1 at k=32. At k=32
that band admits a problem the teacher solves 31 times out of 32, which is not
headroom in any useful sense. This script shows the real distribution and lets
you cut a genuinely hard band out of the probe you already paid for.

  python scripts/00h_reband.py --probed data/probed_all.json
  python scripts/00h_reband.py --probed data/probed_all.json \
      --lo 0.15 --hi 0.70 --out data/trace_targets_hard.json
"""
import argparse
import json
import sys
from collections import Counter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probed", required=True)
    ap.add_argument("--lo", type=float, default=None)
    ap.add_argument("--hi", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--max-n", type=int, default=0,
                    help="cap the written set, hardest first")
    args = ap.parse_args()

    with open(args.probed, "r", encoding="utf-8") as f:
        rows = json.load(f)

    rows = [r for r in rows if r.get("pass_rate") is not None]
    ps = [float(r["pass_rate"]) for r in rows]
    n = len(ps)
    bar = "=" * 66
    print(bar)
    print("PASS-RATE DISTRIBUTION  (%d probed problems)" % n)
    print(bar)

    edges = [0.0, 0.001, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 0.999, 1.01]
    labels = ["p == 0", "0 - .125", ".125-.25", ".25-.375", ".375-.5",
              ".5-.625", ".625-.75", ".75-.875", ".875- <1", "p == 1"]
    hist = Counter()
    for p in ps:
        for i in range(len(edges) - 1):
            if edges[i] <= p < edges[i + 1]:
                hist[i] += 1
                break
    for i, lab in enumerate(labels):
        c = hist.get(i, 0)
        blocks = "#" * int(round(60.0 * c / max(1, n)))
        print("  %-9s %5d  %5.1f%%  %s" % (lab, c, 100.0 * c / n, blocks))

    old = [p for p in ps if 0.0 < p < 1.0]
    if old:
        print("")
        print("  current band 0 < p < 1 : n=%d  mean=%.3f  median=%.3f"
              % (len(old), sum(old) / len(old), sorted(old)[len(old) // 2]))
        hi_share = sum(1 for p in old if p >= 0.75) / len(old)
        print("  of those, p >= 0.75    : %.1f%%  <- near-ceiling, not headroom"
              % (100.0 * hi_share))

    print("")
    print("CANDIDATE BANDS")
    print("  %-16s %6s %7s %9s" % ("band", "n", "mean p", "pass@6 UB"))
    cands = [(0.001, 0.999), (0.05, 0.95), (0.10, 0.90), (0.15, 0.80),
             (0.15, 0.70), (0.20, 0.75), (0.25, 0.75), (0.05, 0.60),
             (0.05, 0.50)]
    for lo, hi in cands:
        sel = [p for p in ps if lo <= p <= hi]
        if not sel:
            continue
        mp = sum(sel) / len(sel)
        ub = sum(1.0 - (1.0 - p) ** 6 for p in sel) / len(sel)
        print("  [%.2f, %.2f]%s %6d %7.3f %9.3f"
              % (lo, hi, " " * 4, len(sel), mp, ub))
    print("")
    print("  Pick a band where pass@6 UB leaves room above a single sample.")
    print("  If pass@6 UB is already ~0.95, SC@6 alone nearly solves the set")
    print("  and no debate can show a gain.")

    if args.lo is None or args.hi is None:
        print(bar)
        print("  Re-run with --lo/--hi/--out to write a target file.")
        return 0

    sel = [r for r in rows if args.lo <= float(r["pass_rate"]) <= args.hi]
    sel.sort(key=lambda r: float(r["pass_rate"]))
    if args.max_n:
        sel = sel[:args.max_n]
    for r in sel:
        r["stratum"] = "hard_band"
    print(bar)
    print("SELECTED  [%.2f, %.2f] -> %d problems" % (args.lo, args.hi, len(sel)))
    if sel:
        q = [float(r["pass_rate"]) for r in sel]
        print("  mean p %.3f   min %.3f   max %.3f"
              % (sum(q) / len(q), min(q), max(q)))
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(sel, f, indent=2)
        print("  wrote %s" % args.out)
    print(bar)
    return 0


if __name__ == "__main__":
    sys.exit(main())
