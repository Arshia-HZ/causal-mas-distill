#!/usr/bin/env python3
"""
Evaluate the Arm A / Arm B checkpoints and decide the hypothesis.

Evaluation protocol (do not change any of this after seeing results):
  - greedy decoding, temperature 0, do_sample False, max_new_tokens 1024
  - grading by eval.grade.is_correct, never a hand-rolled matcher
  - 3 training seeds per arm; a problem's score is the MEAN over its 3 seeds
  - paired bootstrap over PROBLEMS (not over examples)
  - the same table repeated split by teacher difficulty

WHY PAIRED: both arms are evaluated on the identical problem set, so pairing
removes problem difficulty from the variance. An unpaired test on ~30 eval
problems has no chance of resolving a 2-3 point difference.

DECISION RULE -- commit before looking:
  CI entirely above 0        -> deliberation carries transferable signal
  CI spans 0 and |d| < 0.02  -> null; fix the harness (Phase 1) and repeat
  CI entirely below 0        -> transcripts are worse training data; a real
                                finding against the STaR operating assumption

USAGE
-----
  python scripts/12_eval_ab.py \\
      --eval data/eval_problems.json \\
      --arm-a ckpt/arm_a/seed0 ckpt/arm_a/seed1 ckpt/arm_a/seed2 \\
      --arm-b ckpt/arm_b/seed0 ckpt/arm_b/seed1 ckpt/arm_b/seed2 \\
      --probed data/probed_all.json \\
      --out results/ab_eval.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.grade import is_correct  # noqa: E402

BASE = "Qwen/Qwen2.5-1.5B-Instruct"


def paired_bootstrap(a: dict, b: dict, pids: list[str],
                     n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """
    Bootstrap the paired difference b - a by resampling PROBLEMS with
    replacement. a[pid] and b[pid] are per-problem mean accuracies in [0, 1].
    """
    rng = random.Random(seed)
    n = len(pids)
    point = sum(b[p] - a[p] for p in pids) / n if n else 0.0
    draws = []
    for _ in range(n_boot):
        s = [pids[rng.randrange(n)] for _ in range(n)]
        draws.append(sum(b[p] - a[p] for p in s) / n)
    draws.sort()
    lo = draws[int((alpha / 2) * n_boot)]
    hi = draws[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    return point, lo, hi


def run_checkpoint(path: str, problems: list[dict], max_new_tokens: int):
    """Load a LoRA checkpoint, generate greedily, return {pid: 0/1}."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(model, path)
    model.eval()

    scores = {}
    for i, p in enumerate(problems):
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": p["question"]}],
            tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**ids, max_new_tokens=max_new_tokens,
                                 do_sample=False, temperature=None, top_p=None,
                                 pad_token_id=tok.eos_token_id)
        text = tok.decode(out[0][ids["input_ids"].shape[1]:],
                          skip_special_tokens=True)
        scores[p["pid"]] = 1.0 if is_correct(text, p["gold"]) else 0.0
        if (i + 1) % 20 == 0:
            print("  %d/%d" % (i + 1, len(problems)), flush=True)

    del model
    torch.cuda.empty_cache()
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--arm-a", nargs="+", required=True)
    ap.add_argument("--arm-b", nargs="+", required=True)
    ap.add_argument("--probed", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--out", default="results/ab_eval.json")
    args = ap.parse_args()

    problems = json.load(open(args.eval, encoding="utf-8"))
    pids = [p["pid"] for p in problems]
    print("eval problems: %d" % len(pids))

    per_seed = {"A": [], "B": []}
    for arm, paths in (("A", args.arm_a), ("B", args.arm_b)):
        for path in paths:
            print("\n[%s] %s" % (arm, path))
            per_seed[arm].append(run_checkpoint(path, problems, args.max_new_tokens))

    print("\nPER-SEED ACCURACY")
    print("%-4s %-6s %s" % ("arm", "seed", "acc"))
    for arm in ("A", "B"):
        for i, s in enumerate(per_seed[arm]):
            print("%-4s %-6d %.3f" % (arm, i, sum(s.values()) / len(pids)))

    # Average the seeds within each problem BEFORE pairing.
    mean = {
        arm: {p: sum(s[p] for s in per_seed[arm]) / len(per_seed[arm]) for p in pids}
        for arm in ("A", "B")
    }

    def report(subset: list[str], label: str):
        if len(subset) < 5:
            print("\n[%s] only %d problems -- too few to report" % (label, len(subset)))
            return None
        acc_a = sum(mean["A"][p] for p in subset) / len(subset)
        acc_b = sum(mean["B"][p] for p in subset) / len(subset)
        d, lo, hi = paired_bootstrap(mean["A"], mean["B"], subset)
        print("\n[%s]  n=%d" % (label, len(subset)))
        print("  A (rejection sampling) : %.3f" % acc_a)
        print("  B (debate transcripts) : %.3f" % acc_b)
        print("  delta B-A              : %+.3f   95%% CI [%+.3f, %+.3f]" % (d, lo, hi))
        if lo > 0:
            print("  => B WINS. Deliberation carries transferable signal.")
        elif hi < 0:
            print("  => A WINS. Debate transcripts are worse training data.")
        else:
            print("  => NULL. CI spans zero.")
        return {"n": len(subset), "acc_a": acc_a, "acc_b": acc_b,
                "delta": d, "ci": [lo, hi]}

    out = {"pooled": report(pids, "POOLED")}

    if args.probed:
        probed = {p.get("pid"): float(p.get("pass_rate", 0))
                  for p in json.load(open(args.probed, encoding="utf-8"))}
        ceiling = [p for p in pids if probed.get(p, 0.0) >= 1.0]
        hard = [p for p in pids if probed.get(p, 0.0) < 1.0]
        out["teacher_ceiling"] = report(ceiling, "TEACHER CEILING p=1.0")
        out["teacher_nonceiling"] = report(hard, "TEACHER NON-CEILING p<1.0")
        print("\nThe non-ceiling stratum is the one that matters. 80%% of MATH")
        print("is at teacher ceiling and both arms will saturate there.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2)
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
