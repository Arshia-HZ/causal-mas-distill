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

SPEED & RESUME IMPROVEMENTS (2026-08-20)
  - Incremental per-checkpoint caching via --cache <file.jsonl>.
    Each completed checkpoint appends one JSON line. On restart, already-
    evaluated checkpoints are skipped automatically.
  - Batched generation (--batch-size 8). Left-pads prompts and generates
    in parallel. Typical 3-5x speedup on T4.
  - Base model loaded ONCE; LoRA adapters hot-swapped. Saves ~30s and
    3GB download per checkpoint.

USAGE
  python scripts/12_eval_abc.py --eval data/eval_problems.json \\
    --cache results/eval_cache.jsonl \\
    --batch-size 8 \\
    --arm-base base \\
    --arm-a CK/arm_a_eqp/seed0/final CK/arm_a_eqp/seed1/final CK/arm_a_eqp/seed2/final \\
    --arm-b CK/arm_b_eqp/seed0/final CK/arm_b_eqp/seed1/final CK/arm_b_eqp/seed2/final \\
    --arm-c CK/arm_c_eqp/seed0/final CK/arm_c_eqp/seed1/final CK/arm_c_eqp/seed2/final \\
    --probed data/probed_all.json --out results/abc_eval.json
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.grade import is_correct  # noqa: E402

BASE = "Qwen/Qwen2.5-1.5B-Instruct"


# ---------- cache helpers --------------------------------------------------

def load_cache(cache_path: str | None) -> dict[str, dict[str, float]]:
    """Load {checkpoint_key: {pid: score}} from a JSONL cache file."""
    if not cache_path or not Path(cache_path).exists():
        return {}
    out = {}
    for line in open(cache_path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        out[rec["key"]] = rec["scores"]
    return out


def append_cache(cache_path: str, key: str, scores: dict[str, float]):
    """Append one checkpoint result to the JSONL cache."""
    Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"key": key, "scores": scores}) + "\n")


def checkpoint_key(arm: str, path: str) -> str:
    """Unique key for a checkpoint. Normalizes trailing slashes."""
    return f"{arm}::{path.rstrip('/')}"


# ---------- bootstrap ------------------------------------------------------

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


# ---------- batched generation ---------------------------------------------

def run_checkpoint_batched(model, tok, problems, max_new_tokens, batch_size):
    """
    Evaluate all problems using left-padded batched generation.
    Returns {pid: 1.0|0.0}.
    """
    import torch

    scores = {}
    prompts_data = []
    for p in problems:
        prompt = tok.apply_chat_template(
            [{"role": "user", "content": p["question"]}],
            tokenize=False, add_generation_prompt=True)
        prompts_data.append((p["pid"], p["gold"], prompt))

    for batch_start in range(0, len(prompts_data), batch_size):
        batch = prompts_data[batch_start:batch_start + batch_size]
        texts = [b[2] for b in batch]

        # Left-pad for batched generation (decoder-only models need left padding)
        encoded = tok(texts, return_tensors="pt", padding=True,
                      truncation=True, max_length=4096).to(model.device)

        with torch.no_grad():
            out = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                pad_token_id=tok.eos_token_id,
            )

        # Decode only the generated part (strip the prompt)
        prompt_len = encoded["input_ids"].shape[1]
        for j, (pid, gold, _) in enumerate(batch):
            gen_ids = out[j][prompt_len:]
            text = tok.decode(gen_ids, skip_special_tokens=True)
            scores[pid] = 1.0 if is_correct(text, gold) else 0.0

        done = min(batch_start + batch_size, len(prompts_data))
        if done % 50 == 0 or done == len(prompts_data):
            acc_so_far = sum(scores.values()) / max(1, len(scores))
            print("  %d/%d  (running acc: %.3f)" % (done, len(prompts_data), acc_so_far),
                  flush=True)

    return scores


# ---------- main -----------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--arm-base", nargs="*", default=[])
    ap.add_argument("--arm-a", nargs="*", default=[])
    ap.add_argument("--arm-b", nargs="*", default=[])
    ap.add_argument("--arm-c", nargs="*", default=[])
    ap.add_argument("--probed", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Batch size for generation. 8 is safe on T4 16GB. "
                         "Use 1 to match original single-sample behavior.")
    ap.add_argument("--cache", default=None,
                    help="JSONL file for incremental checkpoint caching. "
                         "Already-evaluated checkpoints are skipped on resume.")
    ap.add_argument("--out", default="results/abc_eval.json")
    args = ap.parse_args()

    problems = json.load(open(args.eval, encoding="utf-8"))
    pids = [p["pid"] for p in problems]
    print("eval problems: %d" % len(pids))

    spec = [("BASE", args.arm_base), ("A", args.arm_a),
            ("B", args.arm_b), ("C", args.arm_c)]
    names = {"BASE": "untrained base", "A": "RS solution",
             "B": "debate transcript", "C": "debate final solution"}

    # Count how many checkpoints we actually need to evaluate
    total_ckpts = sum(len(paths) for _, paths in spec if paths)
    if total_ckpts == 0:
        sys.exit("No checkpoints specified. Use --arm-base/--arm-a/--arm-b/--arm-c.")

    # Load cache of already-evaluated checkpoints
    cache = load_cache(args.cache)
    cached_count = 0

    # Load base model ONCE, then hot-swap LoRA adapters
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\nLoading base model: %s" % BASE)
    tok = AutoTokenizer.from_pretrained(BASE)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    # Set left padding for batched generation with decoder-only models
    tok.padding_side = "left"

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.float16, device_map="auto")
    base_model.eval()
    print("Base model loaded.\n")

    per_seed = {}
    t0 = time.time()
    ckpt_idx = 0

    for arm, paths in spec:
        if not paths:
            continue
        per_seed[arm] = []
        for path in paths:
            ckpt_idx += 1
            key = checkpoint_key(arm, path)

            # Check cache first
            if key in cache:
                cached_count += 1
                print("[%s] %s  <- CACHED (skipping)" % (arm, path))
                per_seed[arm].append(cache[key])
                continue

            elapsed = (time.time() - t0) / 60
            print("\n[%s] %s  (%d/%d, t+%.0f min)"
                  % (arm, path, ckpt_idx, total_ckpts, elapsed))

            # Attach LoRA if this is not the base model
            if path and path.lower() != "base":
                from peft import PeftModel
                model = PeftModel.from_pretrained(base_model, path)
                model.eval()
            else:
                model = base_model

            scores = run_checkpoint_batched(
                model, tok, problems, args.max_new_tokens, args.batch_size)

            per_seed[arm].append(scores)

            # Save incrementally
            if args.cache:
                append_cache(args.cache, key, scores)
                print("  -> cached to %s" % args.cache)

            # Unload LoRA adapter (but keep base model)
            if path and path.lower() != "base":
                del model
                torch.cuda.empty_cache()

    # Clean up GPU
    del base_model
    torch.cuda.empty_cache()

    elapsed = (time.time() - t0) / 60
    print("\n\nAll checkpoints evaluated in %.0f min "
          "(%d cached, %d freshly evaluated)"
          % (elapsed, cached_count, ckpt_idx - cached_count))

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
