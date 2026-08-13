#!/usr/bin/env python3
"""
Chapter 1 audit: does the debate actually beat a budget-matched single model?

  arm A  single CoT, 1 sample
  arm B  self-consistency @ n_sc, TOKEN-MATCHED to the debate
  arm C  the multi-agent debate            (requires --traces)

Matching on budget rather than on calls is the whole point. A 3-round debate
spends ~6 generations; compared against 1 sample it wins trivially and the
result means nothing. Compared against 6 samples of self-consistency it often
does not win, and that negative result, measured cleanly and stratified by
headroom, is a genuine contribution rather than a failed experiment.

Three things this script is careful about:

1. n_probe != n_sc. The pass rate wants MANY samples (low variance). The
   self-consistency arm must use exactly the debate budget. Conflating them
   silently hands arm B a budget it never paid for.
2. Single-CoT accuracy is estimated by the pass rate, not by one Bernoulli
   draw. Same expectation, far less variance, free.
3. Debate correctness is averaged over seeds per problem. Keying a dict on pid
   would silently keep whichever seed happened to be last in the file.

Usage:
  python scripts/00d_build_audit_pool.py --probed data/probed_all.json
  python scripts/01_generate_debates.py  --problems data/trace_targets.json
  python scripts/00c_headroom_audit.py --problems data/audit_pool.json \
      --traces data/traces.jsonl --n-probe 32 --n-sc 6
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.backends.api import ApiBackend
from src.debate.prompts import get_solve_prompt
from src.analysis.stats import paired_bootstrap_ci, required_k
from eval.grade import is_correct, extract_answer, _normalize


async def probe_problem(backend, question, gold, n, temperature):
    msgs = [{"role": "user", "content": get_solve_prompt(question)}]
    # No nonce. 00a probed these exact problems with this exact prompt at this
    # exact k/temperature/max_tokens/model, so with the probe cache mounted
    # every one of these is a cache HIT: free, instant, and guaranteed to be
    # the same draws that defined the strata. A nonce would re-randomise and
    # re-bill for nothing.
    outs = await backend.generate(msgs, n=n, temperature=temperature)
    flags = [bool(is_correct(o, gold)) for o in outs]
    answers = [extract_answer(o) for o in outs]
    return (sum(flags) / max(len(flags), 1)), answers, flags


def majority_vote(answers):
    """Vote over NORMALISED answers.

    Voting on raw strings splits the vote between textually different but
    mathematically identical answers ('1/2' vs '0.5' vs '\\dfrac{1}{2}'), which
    understates self-consistency. Since SC is the arm the debate must beat,
    that bias would manufacture a passing gate. Normalise, then vote, then
    return a raw representative for grading.
    """
    buckets = defaultdict(list)
    for a in answers:
        if a:
            buckets[_normalize(str(a))].append(a)
    if not buckets:
        return None
    best = max(buckets.items(), key=lambda kv: len(kv[1]))
    return best[1][0]


def self_consistency_accuracy(answers, gold, n_sc, n_repeats, rng):
    """Expected accuracy of majority-vote@n_sc, estimated by resampling subsets
    of size n_sc from the probe samples. Averaging over subsets removes the
    which-six-did-you-draw lottery from the headline number."""
    if not answers:
        return 0.0
    n_sc = min(n_sc, len(answers))
    hits = 0
    for _ in range(n_repeats):
        subset = rng.sample(answers, n_sc)
        v = majority_vote(subset)
        hits += int(v is not None and is_correct("\\boxed{" + str(v) + "}", gold))
    return hits / n_repeats


def load_debate_accuracy(path):
    """pid -> mean final_correct across all seeds for that pid."""
    # 01_generate_debates.py writes a pretty-printed JSON ARRAY even though the
    # filename ends in .jsonl. Parsing that line by line dies on the opening
    # bracket. Sniff the first non-space character instead of trusting the
    # extension.
    raw = open(path).read()
    stripped = raw.lstrip()
    if stripped.startswith("["):
        traces = json.loads(raw)
    else:
        traces = [json.loads(ln) for ln in raw.splitlines() if ln.strip()]

    per_pid = defaultdict(list)
    n_lines = 0
    for t in traces:
        pid = t.get("pid")
        if pid is None:
            continue
        per_pid[pid].append(1.0 if t.get("final_correct") else 0.0)
        n_lines += 1
    acc = {pid: sum(v) / len(v) for pid, v in per_pid.items()}
    seeds = {pid: len(v) for pid, v in per_pid.items()}
    return acc, seeds, n_lines


def summarise(records, key_a, key_b, label_a, label_b):
    a = [r[key_a] for r in records]
    b = [r[key_b] for r in records]
    ci = paired_bootstrap_ci(a, b)
    return {
        "n": len(records),
        "acc_" + label_a: sum(a) / len(a) if a else 0.0,
        "acc_" + label_b: sum(b) / len(b) if b else 0.0,
        "delta": ci.to_dict(),
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="data/audit_pool.json",
                    help="Stratified pool from 00d. Do NOT pass gate_problems.json: "
                         "it is pre-filtered to headroom and destroys the contrast.")
    ap.add_argument("--config", default="configs/teacher_api_deepseek.yaml")
    ap.add_argument("--model", default=None,
                    help="OVERRIDES the config. Must be the SAME model that "
                         "generated the traces, or the gate compares two "
                         "different models and means nothing.")
    ap.add_argument("--api-url", default=None, help="Overrides config base_url.")
    ap.add_argument("--api-key", default=None, help="Literal key; overrides api_key_env.")
    ap.add_argument("--max-tokens", type=int, default=768,
                    help="Must equal the cap used by 01_generate_debates.py. "
                         "ApiBackend defaults to 768 and the harness passes "
                         "None, so the debate arm ran at 768. Using the "
                         "config's 2048 here hands the baseline 2.7x the "
                         "output budget AND breaks probe-cache reuse.")
    ap.add_argument("--traces", default=None, help="Debate traces jsonl. REQUIRED for arm C.")
    ap.add_argument("--n-probe", type=int, default=32)
    ap.add_argument("--n-sc", type=int, default=6,
                    help="Self-consistency budget. Match the debate generation count.")
    ap.add_argument("--sc-repeats", type=int, default=64)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--concurrency", type=int, default=24)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-path", default="cache_audit.jsonl")
    ap.add_argument("--out", default="results/audit/headroom_audit.json")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    cfg = yaml.safe_load(open(args.config))
    # The repo config nests everything under a top-level `teacher:` key, so
    # cfg["model"] raises KeyError. Accept both the nested and flat shapes.
    cfg = cfg.get("teacher", cfg)
    problems = json.load(open(args.problems))
    if args.limit:
        problems = problems[: args.limit]

    bang = "!" * 66
    if not args.traces:
        print(bang)
        print("WARNING: no --traces given, so there is NO DEBATE ARM.")
        print("You will get SC vs single only, which does not answer the gate.")
        print(bang)

    strata_in = Counter(p.get("stratum", "unlabelled") for p in problems)
    if strata_in.get("ceiling", 0) == 0:
        print(bang)
        print("WARNING: the pool contains no ceiling problems, so")
        print("effective_support_frac will be meaningless. Build the pool with")
        print("scripts/00d_build_audit_pool.py first.")
        print(bang)

    model = args.model or cfg.get("model")
    base_url = args.api_url or cfg.get("base_url")
    print("-" * 66)
    print("arm A/B will be generated with:")
    print("  model      : " + str(model))
    print("  base_url   : " + str(base_url))
    print("  max_tokens : " + str(args.max_tokens))
    print("These MUST match 01_generate_debates.py or the gate is invalid.")
    print("-" * 66)

    backend = ApiBackend(
        model=model,
        base_url=base_url,
        api_key=args.api_key,
        api_key_env=None if args.api_key else cfg.get("api_key_env"),
        max_tokens=args.max_tokens,
        cache_path=args.cache_path,
        concurrency=args.concurrency,
        extra_body={"thinking": {"type": "disabled"}},
    )

    debate_acc, debate_seeds, n_trace_lines = ({}, {}, 0)
    if args.traces:
        debate_acc, debate_seeds, n_trace_lines = load_debate_accuracy(args.traces)
        seed_counts = Counter(debate_seeds.values())
        print("loaded " + str(n_trace_lines) + " traces over " + str(len(debate_acc))
              + " problems (seeds per problem: " + str(dict(seed_counts)) + ")")

    sem = asyncio.Semaphore(args.concurrency)
    done = 0
    total = len(problems)

    async def one(i, p):
        nonlocal done
        pid = p.get("pid") or ("p%04d" % i)
        q = p.get("question") or p.get("problem") or ""
        gold = str(p.get("gold") or p.get("answer") or "")
        if not q or not gold:
            return None
        async with sem:
            rate, answers, flags = await probe_problem(
                backend, q, gold, args.n_probe, args.temperature
            )
        sc_acc = self_consistency_accuracy(answers, gold, args.n_sc, args.sc_repeats, rng)
        done += 1
        if done % 25 == 0 or done == total:
            print("  probed " + str(done) + "/" + str(total), flush=True)
        auto = "ceiling" if rate >= 1.0 else "floor" if rate <= 0.0 else "headroom"
        return {
            "pid": pid,
            "stratum": p.get("stratum") or auto,
            "sample_weight": p.get("sample_weight", 1.0),
            "pass_rate": rate,
            "single": rate,
            "sc": sc_acc,
            "debate": debate_acc.get(pid),
            "n_seeds": debate_seeds.get(pid, 0),
            "headroom": 0.0 < rate < 1.0,
            "n_probe": args.n_probe,
            "n_sc": args.n_sc,
        }

    gathered = await asyncio.gather(*[one(i, p) for i, p in enumerate(problems)])
    records = [r for r in gathered if r]
    await backend.close()

    n = len(records)
    if n == 0:
        print("no usable problems")
        return

    counts = Counter(r["stratum"] for r in records)
    summary = {
        "n_problems": n,
        "strata": dict(counts),
        "effective_support_frac": counts.get("headroom", 0) / n,
        "n_probe": args.n_probe,
        "n_sc": args.n_sc,
        "sc_vs_single": summarise(records, "sc", "single", "sc", "single"),
    }

    withd = [r for r in records if r["debate"] is not None]
    if withd:
        summary["debate_vs_sc"] = summarise(withd, "debate", "sc", "debate", "sc")
        summary["debate_vs_single"] = summarise(withd, "debate", "single", "debate", "single")
        for stratum in ("headroom", "ceiling", "floor"):
            sub = [r for r in withd if r["stratum"] == stratum]
            if len(sub) >= 5:
                summary["debate_vs_sc__" + stratum] = summarise(sub, "debate", "sc", "debate", "sc")
    else:
        summary["debate_vs_sc"] = None

    summary["power_note"] = {
        "k_for_effect_0.10": required_k(0.10),
        "k_for_effect_0.25": required_k(0.25),
        "comment": "Samples per arm for 80% power at alpha=0.05.",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "records": records}, open(out, "w"), indent=2)

    def line(tag, block):
        if not block:
            return
        d = block["delta"]
        star = "  <-- CI excludes 0" if (d["lo"] > 0 or d["hi"] < 0) else ""
        print("  %-26s: %+.3f  [%+.3f, %+.3f]  n=%d%s"
              % (tag, d["point"], d["lo"], d["hi"], block["n"], star))

    bar = "=" * 66
    print("\n" + bar)
    print("HEADROOM AUDIT")
    print(bar)
    print("problems                  : %d" % n)
    for k in ("headroom", "ceiling", "floor"):
        if counts.get(k):
            print("  %-24s: %4d  (%.1f%%)" % (k, counts[k], 100.0 * counts[k] / n))
    print("effective support         : %.1f%%   <- only these problems can move"
          % (100.0 * summary["effective_support_frac"]))
    print("budget                    : n_probe=%d, n_sc=%d" % (args.n_probe, args.n_sc))
    print("-" * 66)
    print("acc single CoT            : %.3f" % summary["sc_vs_single"]["acc_single"])
    print("acc self-consistency@%-4d : %.3f" % (args.n_sc, summary["sc_vs_single"]["acc_sc"]))
    if withd:
        print("acc debate                : %.3f" % summary["debate_vs_sc"]["acc_debate"])
    print("-" * 66)
    print("paired differences (95% bootstrap CI)")
    line("SC - single", summary["sc_vs_single"])
    line("debate - single", summary.get("debate_vs_single"))
    line("debate - SC  [THE GATE]", summary.get("debate_vs_sc"))
    for stratum in ("headroom", "ceiling", "floor"):
        line("debate - SC (" + stratum + ")", summary.get("debate_vs_sc__" + stratum))
    print(bar)
    if withd:
        d = summary["debate_vs_sc"]["delta"]
        if d["lo"] > 0:
            print("GATE PASSED: debate beats matched-budget self-consistency.")
            print("  -> the debate is a legitimate teacher. Proceed to utility")
            print("     estimation on the headroom stratum.")
        else:
            print("GATE NOT PASSED: the CI contains 0 (or is negative).")
            print("  -> the debate is NOT buying accuracy at matched budget.")
            print("     This is a reportable finding, not a failure. The thesis")
            print("     becomes MAS-as-data-engine: debate transcripts are still")
            print("     valuable as TRAINING DATA (error-correction traces that")
            print("     self-consistency cannot produce) even when they are not")
            print("     worth the tokens at inference time.")
    else:
        print("NO DEBATE ARM. Re-run with --traces to answer the gate.")
    print(bar)
    print("written -> " + str(out))


if __name__ == "__main__":
    asyncio.run(main())
