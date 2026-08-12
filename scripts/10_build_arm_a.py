#!/usr/bin/env python3
"""
Build the Arm A (rejection sampling / STaR) pool from the EXISTING probe cache.
Zero API calls.

WHY THIS IS FREE
----------------
scripts/00a_probe_difficulty.py already sampled the teacher 32 times for each
of 785 problems to measure difficulty. Those completions are rejection samples.
They are sitting in cache_probe.jsonl. Recover them instead of paying again.

CRITICAL -- PROMPT VERSION LOCK
-------------------------------
The cache key is a hash of the exact prompt string. The probe ran under the v1
solve prompt. If you install the v3 prompts.py, get_solve_prompt() returns a
DIFFERENT string and every cache lookup misses silently, and this script will
report 0% recovery and you will think the cache is gone.

So the v1 prompt is hardcoded below as LEGACY_SOLVE_PROMPT. Do not "clean this
up" by importing from prompts.py.

Note the v1 placeholder is {problem}, not {question}.

USAGE
-----
  python scripts/10_build_arm_a.py \\
      --cache-path /content/drive/MyDrive/cmd/cache_probe.jsonl \\
      --problems data/probed_all.json \\
      --out data/arm_a_pool.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.grade import is_correct  # noqa: E402

# --- v1 prompt, frozen. Must match what 00a actually sent. -----------------
LEGACY_SOLVE_PROMPT = """Solve the following problem step by step. Show your reasoning:

{problem}
"""


def legacy_solve_prompt(question: str) -> str:
    return LEGACY_SOLVE_PROMPT.format(problem=question)


def key_of(payload) -> str:
    """Identical to src/backends/api.py::key_of."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:32]


def load_cache(path: Path):
    cache, bad = {}, 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
                cache[r["k"]] = r["v"]
            except (json.JSONDecodeError, KeyError):
                bad += 1
    return cache, bad


def chunk_keys(messages, n, temperature, max_tokens, model,
               max_n_per_request=8, nonce=None):
    """
    Reproduce ApiBackend.generate's chunking. A request for n=32 with a cap of
    8 per request becomes 4 cached entries whose nonce is '|chunk{i}'.
    A single-chunk request uses the bare nonce.
    """
    chunks, remaining = [], n
    while remaining > 0:
        take = min(remaining, max_n_per_request)
        chunks.append(take)
        remaining -= take
    keys = []
    for ci, take in enumerate(chunks):
        c = nonce if len(chunks) == 1 else "%s|chunk%d" % (nonce or "", ci)
        keys.append(key_of({"m": messages, "n": take, "t": temperature,
                            "mt": max_tokens, "mo": model, "c": c}))
    return keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-path", required=True)
    ap.add_argument("--problems", required=True,
                    help="data/probed_all.json (all 785, with pass_rate)")
    ap.add_argument("--model", default="deepseek-v3.2")
    ap.add_argument("--n-probe", type=int, default=32)
    ap.add_argument("--max-tokens", type=int, default=768,
                    help="What the probe actually used. NOT 1024.")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-n-per-request", type=int, default=8)
    ap.add_argument("--out", default="data/arm_a_pool.jsonl")
    args = ap.parse_args()

    cache, bad = load_cache(Path(args.cache_path))
    problems = json.load(open(args.problems, encoding="utf-8"))
    print("cache entries : %d%s" % (len(cache), "  (%d bad lines)" % bad if bad else ""))
    print("problems      : %d" % len(problems))

    full, partial, missing = 0, 0, 0
    n_out, n_with_correct, total_correct = 0, 0, 0

    with open(args.out, "w", encoding="utf-8") as fh:
        for p in problems:
            q = p.get("question") or p.get("problem") or ""
            gold = p.get("gold") or p.get("answer") or ""
            pid = p.get("pid") or p.get("id")
            msgs = [{"role": "user", "content": legacy_solve_prompt(q)}]
            keys = chunk_keys(msgs, args.n_probe, args.temperature,
                              args.max_tokens, args.model,
                              args.max_n_per_request)
            hits = [k for k in keys if k in cache]
            if not hits:
                missing += 1
                continue
            full += (len(hits) == len(keys))
            partial += (0 < len(hits) < len(keys))

            samples = []
            for k in hits:
                samples.extend(cache[k])

            correct = [s for s in samples if s and is_correct(s, gold)]
            total_correct += len(correct)
            n_out += 1
            if correct:
                n_with_correct += 1
            fh.write(json.dumps({
                "pid": pid, "question": q, "gold": gold,
                "n_sampled": len(samples),
                "solutions": correct,
            }, ensure_ascii=False) + "\n")

    total = len(problems)
    rate = (full + partial) / total if total else 0.0
    print("\nRECOVERY")
    print("  all chunks hit  : %d" % full)
    print("  partial hit     : %d" % partial)
    print("  no hit          : %d" % missing)
    print("  recovery rate   : %.1f%%" % (100 * rate))
    print("\nARM A POOL -> %s" % args.out)
    print("  problems written        : %d" % n_out)
    print("  with >=1 correct sample : %d" % n_with_correct)
    print("  total correct solutions : %d" % total_correct)

    if rate < 0.90:
        print("\nSTOP. Recovery below 90%. Do NOT fall back to the API silently.")
        print("Most likely cause: --max-tokens or the prompt string does not")
        print("match what 00a actually sent. Try --max-tokens 768 and 1024, and")
        print("confirm LEGACY_SOLVE_PROMPT is byte-identical to the v1 string.")
        sys.exit(2)


if __name__ == "__main__":
    main()
