#!/usr/bin/env python3
"""
Evaluate Arm A / B / C plus an UNTRAINED BASE control.

The base control is not optional. On 2026-08-13 both arms scored 0.156 and
0.125, which is at or BELOW an untrained Qwen2.5-1.5B-Instruct on MATH.
Without a base row you cannot tell "B lost to A" from "both runs damaged the
model". Pass --arm-base base to add it.

Protocol, fixed before looking at results:
  greedy decoding, max_new_tokens 1024, eval.grade.is_correct only,
  3 seeds per arm, per-problem score = mean over seeds,
  paired bootstrap over PROBLEMS, every arm compared against A.

USAGE
  python scripts/12_eval_abc.py --eval data/eval_problems.json \\
    --arm-base base \\
    --arm-a CK/arm_a_eqp/seed0/final CK/arm_a_eqp/seed1/final \\
    --arm-b CK/arm_b_eqp/seed0/final CK/arm_b_eqp/seed1/final \\
    --arm-c CK/arm_c_eqp/seed0/final CK/arm_c_eqp/seed1/final \\
    --probed data/probed_all.json --out results/abc_eval.json
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


def paired_bootstrap(a, b, pids, n_boot=10000, alpha=0.05, seed=0):
    rng = random.Random(seed)
    n = len(pids)
    if not n:
        return 0.0, 0.0, 0.0
    point = sum(b[p] - a[p] for p in pids) / n
    draws = []
    for _ in range(n_boot):
        s = [pids[rng.randrange(n)] for _ in range(n)]
        draws.append(sum(b[p] - a[p] for p in s) / n)
    draws.sort()
    lo = draws[int((alpha / 2) * n_boot)]
    hi = draws[min(int((1 - alpha / 2) * n_boot), n_boot - 1)]
    return point, lo, hi


def run_checkpoint(path, problems, max_new_tokens):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE)
    model = AutoModelForCausalLM.from_pretrained(
        BASE, dtype=torch.float16, device_map="auto")
    if path and path.lower() != "base":
        from peft import PeftModel
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
        if (i + 1) % 50 == 0:
            print("  %d/%d" % (i + 1, len(problems)), flush=True)

    del model
    torch.cuda.empty_cache()
    return scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--arm-base", nargs="*", default=[])
    ap.add_argument("--arm-a", nargs="*", default=[])
    ap.add_argument("--arm-b", nargs="*", default=[])
    ap.add_argument("--arm-c", nargs="*", default=[])
    ap.add_argument("--probed", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--out", default="results/abc_eval.json")
    args = ap.parse_args()

    problems = json.load(open(args.eval, encoding="utf-8"))
    pids = [p["pid"] for p in problems]
    print("eval problems: %d" % len(pids))

    spec = [("BASE", args.arm_base), ("A", args.arm_a),
            ("B", args.arm_b), ("C", args.arm_c)]
    names = {"BASE": "untrained base", "A": "RS solution",
             "B": "debate transcript", "C": "debate final solution"}

    per_seed = {}
    for arm, paths in spec:
        if not paths:
            continue
        per_seed[arm] = []
        for path in paths:
            print("\n[%s] %s" % (arm, path))
            per_seed[arm].append(run_checkpoint(path, problems, args.max_new_tokens))

    if "A" not in per_seed:
        sys.exit("--arm-a is required as the reference arm")

    print("\nPER-SEED ACCURACY")
    for arm in per_seed:
        for i, s in enumerate(per_seed[arm]):
            print("  %-5s seed%d  %.3f" % (arm, i, sum(s.values()) / len(pids)))

    mean = {arm: {p: sum(s[p] for s in ss) / len(ss) for p in pids}
            for arm, ss in per_seed.items()}

    def report(subset, label):
        if len(subset) < 5:
            print("\n[%s] only %d problems -- too few" % (label, len(subset)))
            return None
        print("\n[%s]  n=%d" % (label, len(subset)))
        res = {"n": len(subset), "arms": {}}
        for arm in mean:
            acc = sum(mean[arm][p] for p in subset) / len(subset)
            res["arms"][arm] = {"acc": acc, "name": names[arm]}
            print("  %-5s %-24s %.3f" % (arm, names[arm], acc))
        for arm in mean:
            if arm == "A":
                continue
            d, lo, hi = paired_bootstrap(mean["A"], mean[arm], subset)
            res["arms"][arm]["delta_vs_a"] = [d, lo, hi]
            verdict = ("WINS" if lo > 0 else "LOSES" if hi < 0 else "NULL")
            print("  delta %s-A  %+.3f  95%% CI [%+.3f, %+.3f]  => %s"
                  % (arm, d, lo, hi, verdict))
        if "C" in mean and "B" in mean:
            d, lo, hi = paired_bootstrap(mean["C"], mean["B"], subset)
            res["b_vs_c"] = [d, lo, hi]
            print("  delta B-C  %+.3f  95%% CI [%+.3f, %+.3f]"
                  "   (is the PROCESS TEXT teachable, beyond the solution?)"
                  % (d, lo, hi))
        return res

    out = {"pooled": report(pids, "POOLED")}

    shared = [p["pid"] for p in problems if p.get("in_shared")]
    if 5 <= len(shared) < len(pids):
        out["in_training_distribution"] = report(shared, "HELD OUT, SAME POOL")
        held = [p["pid"] for p in problems if not p.get("in_shared")]
        out["out_of_pool"] = report(held, "OUT OF POOL")

    if args.probed:
        pr = {p.get("pid"): float(p.get("pass_rate", 0))
              for p in json.load(open(args.probed, encoding="utf-8"))}
        out["teacher_ceiling"] = report(
            [p for p in pids if pr.get(p, 0) >= 1.0], "TEACHER CEILING")
        out["teacher_nonceiling"] = report(
            [p for p in pids if pr.get(p, 0) < 1.0], "TEACHER NON-CEILING")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2)
    print("\nwrote %s" % args.out)
    print("\nREAD THE BASE ROW FIRST. If A and B are both below BASE, the "
          "training pipeline is broken, not the hypothesis.")


if __name__ == "__main__":
    main()
