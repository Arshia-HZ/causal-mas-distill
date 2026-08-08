#!/usr/bin/env python3
"""
Build a STRATIFIED audit pool.

Why this exists
---------------
`data/gate_problems.json` is already filtered to the mid-difficulty band
(0 < pass_rate < 1). Feeding that file to `00c_headroom_audit.py` destroys the
comparison the audit is supposed to make: every problem is headroom, so
`effective_support_frac` comes out at 1.0 and the population-level claim
("86% of MATH has no headroom") cannot be computed.

The audit needs BOTH strata in one pool so the contrast is measured on the same
run, with the same sampler, at the same temperature.

Recovering the ceiling problems
-------------------------------
`00a_probe_difficulty.py` only writes the *kept* problems. To recover the full
labelled set, re-run the probe with the filter opened up. Point it at the same
`--cache-path` and every request is a cache hit, so it costs nothing and
finishes in seconds:

    python scripts/00a_probe_difficulty.py \\
        --input data/math_problems.json \\
        --output data/probed_all.json \\
        --api-url $DEEPSEEK_URL --api-key $DEEPSEEK_API_KEY \\
        --k 8 --keep-min 0.0 --keep-max 1.0 --max-problems 100000 \\
        --cache-path cache_probe.jsonl

Then:

    python scripts/00d_build_audit_pool.py \\
        --probed data/probed_all.json --out data/audit_pool.json
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


def stratum_of(pass_rate: float | None) -> str:
    if pass_rate is None:
        return "unknown"
    if pass_rate >= 1.0:
        return "ceiling"
    if pass_rate <= 0.0:
        return "floor"
    return "headroom"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probed", required=True,
                    help="All probed problems WITH pass_rate (unfiltered).")
    ap.add_argument("--gate", default=None,
                    help="Optional data/gate_problems.json to union in.")
    ap.add_argument("--n-ceiling", type=int, default=150,
                    help="Ceiling problems to sample. Enough to estimate the "
                         "population fraction, not so many that arm C costs a day.")
    ap.add_argument("--n-floor", type=int, default=40)
    ap.add_argument("--n-headroom", type=int, default=None,
                    help="Default: keep all of them. They are the scarce resource.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/audit_pool.json")
    ap.add_argument("--out-traces", default="data/trace_targets.json",
                    help="Headroom subset only; generate debates for these.")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    probed = json.load(open(args.probed))

    by_pid: dict[str, dict] = {}
    for i, p in enumerate(probed):
        pid = p.get("pid") or f"p{i:04d}"
        p = dict(p)
        p["pid"] = pid
        p["stratum"] = stratum_of(p.get("pass_rate"))
        by_pid[pid] = p

    if args.gate:
        for i, p in enumerate(json.load(open(args.gate))):
            pid = p.get("pid") or f"gate{i:04d}"
            if pid not in by_pid:
                p = dict(p)
                p["pid"] = pid
                p["stratum"] = stratum_of(p.get("pass_rate"))
                by_pid[pid] = p

    buckets: dict[str, list[dict]] = {"headroom": [], "ceiling": [], "floor": [], "unknown": []}
    for p in by_pid.values():
        buckets[p["stratum"]].append(p)
    for v in buckets.values():
        v.sort(key=lambda p: p["pid"])

    def take(name: str, n: int | None) -> list[dict]:
        pool = buckets[name]
        if n is None or n >= len(pool):
            return list(pool)
        return rng.sample(pool, n)

    headroom = take("headroom", args.n_headroom)
    selected = headroom + take("ceiling", args.n_ceiling) + take("floor", args.n_floor)
    rng.shuffle(selected)

    # sampling weights let you reweight back to the population if you want the
    # population-level accuracy rather than the pool-level accuracy
    counts = {k: len(v) for k, v in buckets.items()}
    for p in selected:
        s = p["stratum"]
        n_sel = sum(1 for q in selected if q["stratum"] == s)
        p["population_count"] = counts[s]
        p["sample_weight"] = counts[s] / n_sel if n_sel else 0.0

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(selected, open(out, "w"), indent=2)

    tout = Path(args.out_traces)
    tout.parent.mkdir(parents=True, exist_ok=True)
    json.dump(headroom, open(tout, "w"), indent=2)

    total = sum(counts.values())
    print("=" * 60)
    print("POPULATION (from the probe)")
    for k in ("headroom", "ceiling", "floor", "unknown"):
        if counts[k]:
            print(f"  {k:<9}: {counts[k]:>5}  ({counts[k]/total:.1%})")
    print(f"  {'total':<9}: {total:>5}")
    print("-" * 60)
    print("AUDIT POOL")
    sel_counts: dict[str, int] = {}
    for p in selected:
        sel_counts[p["stratum"]] = sel_counts.get(p["stratum"], 0) + 1
    for k, v in sorted(sel_counts.items()):
        print(f"  {k:<9}: {v:>5}")
    print(f"  {'total':<9}: {len(selected):>5}  -> {out}")
    print(f"  headroom subset for debate generation -> {tout} ({len(headroom)})")
    print("=" * 60)
    print("Next: generate debates for the headroom subset, then run the audit")
    print("WITH --traces. Without --traces there is no debate arm and the audit")
    print("answers nothing.")


if __name__ == "__main__":
    main()
