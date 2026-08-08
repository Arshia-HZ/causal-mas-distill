#!/usr/bin/env python3
"""
Chapter 1 audit: does the debate actually beat a budget-matched single model?

Run this BEFORE any more utility estimation. It answers the question that
makes or breaks the whole thesis, and it is cheap.

Your probe already showed 676/785 MATH problems at teacher pass-rate 1.0. On
those problems a debate cannot help, because there is nothing to fix. Any
aggregate "MAS beats single-agent" number computed over that pool is dominated
by problems with zero headroom. This script measures the comparison properly:

  arm A  single CoT, 1 sample                (cheapest)
  arm B  self-consistency @ n, TOKEN-MATCHED to the debate
  arm C  the multi-agent debate              (needs --traces)

Matching on tokens rather than on calls is the whole point. A 3-round debate
spends roughly 6 generations; compared against 1 sample it wins trivially and
the result means nothing. Compared against 6 samples of self-consistency it
often does not win -- and that negative result, measured cleanly and
stratified by headroom, is a genuine contribution rather than a failed
experiment.

Outputs
  results/audit/headroom_audit.json   per-problem records + aggregate CIs
  stdout                              a table you can paste into the thesis

Usage
  python scripts/00c_headroom_audit.py --problems data/gate_problems.json \\
      --config configs/teacher_api_deepseek.yaml --n-probe 8 --limit 60
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import yaml

from src.backends.api import ApiBackend
from src.debate.prompts import get_solve_prompt
from src.analysis.stats import paired_bootstrap_ci, required_k
from eval.grade import is_correct, extract_answer


async def probe_problem(backend, question, gold, n, temperature):
    msgs = [{"role": "user", "content": get_solve_prompt(question)}]
    outs = await backend.generate(msgs, n=n, temperature=temperature, cache_nonce="audit")
    flags = [bool(is_correct(o, gold)) for o in outs]
    answers = [extract_answer(o) for o in outs]
    return (sum(flags) / max(len(flags), 1)), answers, flags


def majority_vote(answers):
    vals = [a for a in answers if a]
    if not vals:
        return None
    return Counter(vals).most_common(1)[0][0]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", default="data/gate_problems.json")
    ap.add_argument("--config", default="configs/teacher_api_deepseek.yaml")
    ap.add_argument("--traces", default=None, help="Optional traces jsonl, adds arm C.")
    ap.add_argument("--n-probe", type=int, default=8, help="6 matches a 3-round debate.")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--out", default="results/audit/headroom_audit.json")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    problems = json.load(open(args.problems))
    if args.limit:
        problems = problems[: args.limit]

    backend = ApiBackend(
        model=cfg["model"],
        base_url=cfg.get("base_url"),
        api_key_env=cfg.get("api_key_env"),
        max_tokens=cfg.get("max_tokens", 2048),
    )

    debate_correct = {}
    if args.traces:
        for line in open(args.traces):
            if line.strip():
                t = json.loads(line)
                debate_correct[t["pid"]] = bool(t.get("final_correct", False))

    records = []
    for i, p in enumerate(problems):
        pid = p.get("pid", "p%04d" % i)
        q = p.get("question") or p.get("problem") or ""
        gold = str(p.get("gold") or p.get("answer") or "")
        if not q or not gold:
            continue

        rate, answers, flags = await probe_problem(backend, q, gold, args.n_probe, args.temperature)
        sc = majority_vote(answers)
        sc_correct = bool(sc is not None and is_correct("\\boxed{" + str(sc) + "}", gold))

        records.append({
            "pid": pid,
            "pass_rate": rate,
            "single_correct": bool(flags[0]) if flags else False,
            "sc_correct": sc_correct,
            "debate_correct": debate_correct.get(pid),
            "headroom": 0.0 < rate < 1.0,
            "n_probe": args.n_probe,
        })
        print("[%d/%d] %s  pass=%.3f  single=%s  sc=%s"
              % (i + 1, len(problems), pid, rate, records[-1]["single_correct"], sc_correct),
              flush=True)

    await backend.close()

    n = len(records)
    ceiling = sum(1 for r in records if r["pass_rate"] >= 1.0)
    floor = sum(1 for r in records if r["pass_rate"] <= 0.0)
    mid = n - ceiling - floor

    single = [1.0 if r["single_correct"] else 0.0 for r in records]
    scv = [1.0 if r["sc_correct"] else 0.0 for r in records]

    summary = {
        "n_problems": n,
        "n_ceiling": ceiling,
        "n_floor": floor,
        "n_headroom": mid,
        "effective_support_frac": (mid / n) if n else 0.0,
        "acc_single": (sum(single) / n) if n else 0.0,
        "acc_self_consistency": (sum(scv) / n) if n else 0.0,
        "sc_vs_single": paired_bootstrap_ci(scv, single).to_dict(),
    }

    have = [r for r in records if r["debate_correct"] is not None]
    if have:
        d = [1.0 if r["debate_correct"] else 0.0 for r in have]
        s = [1.0 if r["sc_correct"] else 0.0 for r in have]
        o = [1.0 if r["single_correct"] else 0.0 for r in have]
        summary["acc_debate"] = sum(d) / len(d)
        summary["debate_vs_sc"] = paired_bootstrap_ci(d, s).to_dict()
        summary["debate_vs_single"] = paired_bootstrap_ci(d, o).to_dict()

        hr = [r for r in have if r["headroom"]]
        if hr:
            dh = [1.0 if r["debate_correct"] else 0.0 for r in hr]
            sh = [1.0 if r["sc_correct"] else 0.0 for r in hr]
            summary["headroom_only"] = {
                "n": len(hr),
                "acc_debate": sum(dh) / len(dh),
                "acc_self_consistency": sum(sh) / len(sh),
                "debate_vs_sc": paired_bootstrap_ci(dh, sh).to_dict(),
            }

    summary["power_note"] = {
        "k_for_effect_0.10": required_k(0.10),
        "k_for_effect_0.25": required_k(0.25),
        "comment": "Samples per arm for 80% power at alpha=0.05.",
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump({"summary": summary, "records": records}, open(out, "w"), indent=2)

    bar = "=" * 66
    print("\n" + bar)
    print("HEADROOM AUDIT")
    print(bar)
    if n:
        print("problems              : %d" % n)
        print("  ceiling (pass=1.0)  : %d  (%.1f%%)" % (ceiling, 100.0 * ceiling / n))
        print("  floor   (pass=0.0)  : %d  (%.1f%%)" % (floor, 100.0 * floor / n))
        print("  headroom            : %d  (%.1f%%)  <- only these can move" % (mid, 100.0 * mid / n))
    print("acc single CoT        : %.3f" % summary["acc_single"])
    print("acc self-consistency  : %.3f" % summary["acc_self_consistency"])
    ci = summary["sc_vs_single"]
    print("  SC - single         : %+.3f  [%+.3f, %+.3f]" % (ci["point"], ci["lo"], ci["hi"]))
    if "acc_debate" in summary:
        print("acc debate            : %.3f" % summary["acc_debate"])
        ci = summary["debate_vs_sc"]
        print("  debate - SC         : %+.3f  [%+.3f, %+.3f]" % (ci["point"], ci["lo"], ci["hi"]))
        print("  ^ if this CI contains 0, the debate is not buying accuracy at")
        print("    matched budget. That is a reportable finding, and it points")
        print("    the thesis at MAS-as-data-engine.")
    print(bar)
    print("written -> %s" % out)


if __name__ == "__main__":
    asyncio.run(main())
