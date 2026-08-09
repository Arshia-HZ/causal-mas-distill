#!/usr/bin/env python3
"""
Rebuild the FULL labelled probe set from the on-disk cache. Zero API calls.

Why this is needed
------------------
00a_probe_difficulty.py writes only the problems it KEPT:

    kept = [p for p in valid if keep(p, args.keep_min, args.keep_max)]
    json.dump(kept, ...)

So data/gate_problems.json holds your 83 headroom problems and nothing else.
The 676 ceiling and 26 floor problems were discarded at write time. 00d needs
them: without a ceiling stratum there is no headroom contrast, and the Chapter 1
result evaporates.

The generations themselves are still sitting in cache_probe.jsonl. This script
recomputes the cache keys, grades the cached samples, and writes probed_all.json
with a pass_rate for every problem it can recover.

The cache key depends on (messages, n, temperature, max_tokens, model, nonce).
If you don't remember exactly what you passed to 00a, run with --auto and it
will grid-search the plausible combinations and report the hit rate for each.

Usage
-----
  python scripts/00e_recover_probe_offline.py \\
      --input data/math_problems.json \\
      --cache-path cache_probe.jsonl \\
      --gate data/gate_problems.json \\
      --auto

If coverage is 100%, go straight to 00d. If it is partial, run 00a normally --
the recovered entries still hit the cache and only the gaps cost API calls.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from src.backends.api import key_of
except Exception:  # openai not installed locally; key_of is pure stdlib
    def key_of(payload) -> str:
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
        ).hexdigest()[:32]

from src.debate.prompts import get_solve_prompt
from eval.grade import is_correct


def load_cache(path):
    cache = {}
    bad = 0
    with open(path) as f:
        for line in f:
            try:
                r = json.loads(line)
                cache[r["k"]] = r["v"]
            except (json.JSONDecodeError, KeyError):
                bad += 1
    return cache, bad


def make_key(question, n, temperature, max_tokens, model, nonce=None):
    messages = [{"role": "user", "content": get_solve_prompt(question)}]
    return key_of({"m": messages, "n": n, "t": temperature,
                   "mt": max_tokens, "mo": model, "c": nonce})


def hit_rate(problems, cache, n, temperature, max_tokens, model, sample=25):
    probe = problems[:sample]
    hits = 0
    for p in probe:
        q = p.get("question") or p.get("problem") or ""
        if make_key(q, n, temperature, max_tokens, model) in cache:
            hits += 1
    return hits / max(len(probe), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True,
                    help="The SAME problems file you originally passed to 00a.")
    ap.add_argument("--cache-path", default="cache_probe.jsonl")
    ap.add_argument("--gate", default=None,
                    help="data/gate_problems.json. Its stored pass_rates are "
                         "trusted and merged in, so the 83 survive even if the "
                         "cache is incomplete.")
    ap.add_argument("--model", default="deepseek-v3.2")
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--max-tokens", type=int, default=768,
                    help="ApiBackend default is 768. 00a did not override it.")
    ap.add_argument("--temperature", type=float, default=0.7,
                    help="00a hardcodes 0.7.")
    ap.add_argument("--auto", action="store_true",
                    help="Grid-search model/max_tokens/k to find the combo that hits.")
    ap.add_argument("--out", default="data/probed_all.json")
    args = ap.parse_args()

    cache_path = Path(args.cache_path)
    if not cache_path.exists():
        print("NO CACHE at %s" % cache_path)
        print("Nothing to recover offline. Re-run 00a for real:")
        print("  --keep-min 0.0 --keep-max 1.0 --max-problems 100000")
        print("At k=8 over ~785 problems that is ~785 requests. Not free, but")
        print("cheap, and it is a one-off.")
        sys.exit(2)

    cache, bad = load_cache(cache_path)
    problems = json.load(open(args.input))
    print("cache entries : %d%s" % (len(cache), ("  (%d unparseable lines)" % bad) if bad else ""))
    print("problems      : %d" % len(problems))

    model, k, mt = args.model, args.k, args.max_tokens

    if args.auto:
        print("\nprobing cache-key combinations on the first 25 problems:")
        models = [args.model, "deepseek-v3.2", "deepseek-v4-flash",
                  "deepseek-chat", "deepseek-reasoner"]
        seen = set()
        models = [m for m in models if not (m in seen or seen.add(m))]
        best = (0.0, model, k, mt)
        for m in models:
            for tok in dict.fromkeys([args.max_tokens, 768, 2048]):
                for kk in dict.fromkeys([args.k, 8]):
                    r = hit_rate(problems, cache, kk, args.temperature, tok, m)
                    if r > 0:
                        print("  model=%-20s max_tokens=%-5d k=%-3d -> %.0f%% hit"
                              % (m, tok, kk, 100 * r))
                    if r > best[0]:
                        best = (r, m, kk, tok)
        if best[0] == 0.0:
            print("  no combination hit the cache.")
            print("\nThe cache was written with different settings, or with a")
            print("different --input file (the prompt text is part of the key).")
            print("Fall back to re-running 00a with the filter opened up.")
            sys.exit(3)
        _, model, k, mt = best
        print("\nusing model=%s k=%d max_tokens=%d" % (model, k, mt))

    gate_rates = {}
    if args.gate and Path(args.gate).exists():
        for p in json.load(open(args.gate)):
            if p.get("pid") and p.get("pass_rate") is not None:
                gate_rates[p["pid"]] = p["pass_rate"]
        print("trusted pass_rates merged from --gate: %d" % len(gate_rates))

    recovered, missing, from_gate = [], [], 0
    for i, p in enumerate(problems):
        p = dict(p)
        pid = p.get("pid") or ("p%04d" % i)
        p["pid"] = pid
        q = p.get("question") or p.get("problem") or ""
        gold = str(p.get("gold") or p.get("answer") or "")

        key = make_key(q, k, args.temperature, mt, model)
        samples = cache.get(key)

        if samples:
            hits = sum(1 for s in samples if is_correct(s, gold))
            p["pass_rate"] = hits / len(samples)
            p["n_samples"] = len(samples)
            p["source"] = "cache"
            recovered.append(p)
        elif pid in gate_rates:
            p["pass_rate"] = gate_rates[pid]
            p["source"] = "gate_file"
            from_gate += 1
            recovered.append(p)
        else:
            missing.append(pid)

    dist = {"floor(0)": 0, "mid": 0, "ceiling(1)": 0}
    for p in recovered:
        r = p["pass_rate"]
        dist["floor(0)" if r == 0 else "ceiling(1)" if r == 1 else "mid"] += 1

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(recovered, open(out, "w"), indent=2)

    bar = "=" * 60
    print("\n" + bar)
    print("OFFLINE RECOVERY")
    print(bar)
    print("recovered   : %d / %d  (%.1f%%)"
          % (len(recovered), len(problems), 100.0 * len(recovered) / max(len(problems), 1)))
    print("  from cache: %d" % (len(recovered) - from_gate))
    print("  from gate : %d" % from_gate)
    print("missing     : %d" % len(missing))
    print("-" * 60)
    for kk, v in dist.items():
        print("  %-11s: %5d" % (kk, v))
    print(bar)
    print("written -> %s" % out)

    if dist["ceiling(1)"] == 0:
        print("\nWARNING: zero ceiling problems recovered. 00d cannot build a")
        print("stratified pool from this. Re-run 00a with the filter opened up.")
    elif missing:
        print("\n%d problems still missing. Either accept the partial set (the" % len(missing))
        print("strata proportions are unbiased if the misses are random), or run")
        print("00a normally -- recovered entries hit the cache, only gaps cost calls.")
    else:
        print("\nFull coverage. Go straight to:")
        print("  python scripts/00d_build_audit_pool.py --probed %s \\" % out)
        print("      --gate %s" % (args.gate or "data/gate_problems.json"))


if __name__ == "__main__":
    main()
