#!/usr/bin/env python3
"""
Audit the provider before spending money on regeneration. Two modes.

--mode limits   What is the REAL context window, and is max_tokens honoured?
                One oversized request per model; the provider's 400 error
                states the true limit. Costs almost nothing.
                Resolves two open mysteries:
                  * the dashboard says 32k but math_0498 died at
                    "Input is 8357 tokens but this model only supports 8192"
                  * 47% of solver messages exceeded the --max-tokens cap,
                    so the cap may never have reached the provider

--mode critic   Which model should play the CRITIC? Feeds each candidate a
                set of round-1 solutions already known to be WRONG and a set
                already known to be RIGHT, then measures:
                  recall    = flagged | solution is wrong   (want high)
                  fpr       = flagged | solution is right   (want low)
                deepseek-v3.2 critiquing itself scores recall 0.10. Any
                candidate that cannot beat that is not worth regenerating with.
                ~40 calls per model. Cents.

USAGE
  python scripts/00j_provider_audit.py --mode limits \\
     --models deepseek-v3.2 deepseek-v3.1 gpt-oss-120b gemma-4-31B-it

  python scripts/00j_provider_audit.py --mode critic \\
     --traces data/traces.jsonl --n-wrong 20 --n-right 20 \\
     --models deepseek-v3.2 deepseek-v3.1 gpt-oss-120b gemma-4-31B-it
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backends.multikey import MultiKeyApiBackend  # noqa: E402

try:
    from eval.grade import is_correct
except Exception:
    def is_correct(pred, gold):
        return (pred or "").strip() == (gold or "").strip()

BASE_URL = "https://api.generalcompute.com/v1"

CRITIC_PROMPT = """You are checking another person's solution to a maths problem.

PROBLEM
{question}

PROPOSED SOLUTION
{solution}

Check every step. Do not re-solve the problem from scratch; audit what is
written. An arithmetic slip, a dropped case, a wrong sign, or an unjustified
leap all count as errors.

Your reply MUST begin with exactly one of these two lines:
VERDICT: ERROR
VERDICT: CORRECT

Then, in at most 120 words, justify the verdict. If ERROR, name the first
step that is wrong and say why."""


def load_any(path):
    text = open(path, encoding="utf-8").read().strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def mk(model, cache, max_tokens=512):
    return MultiKeyApiBackend(
        model=model, base_url=BASE_URL, cache_path=cache,
        concurrency_per_key=4, max_tokens=max_tokens,
        extra_body={"thinking": {"type": "disabled"}})


async def probe_limits(models, cache, probe_chars):
    print("%-20s %14s %14s %s" % ("model", "real_ctx", "max_tok_ok", "note"))
    rows = []
    for m in models:
        be = mk(m, cache, max_tokens=64)
        ctx, note = "?", ""
        filler = ("The quick brown fox jumps over the lazy dog. "
                  * (probe_chars // 45))
        try:
            await be.generate([{"role": "user", "content": filler + "\nSay OK."}],
                              n=1, temperature=0.0, max_tokens=8,
                              cache_nonce="ctxprobe")
            ctx = ">%d chars" % len(filler)
            note = "no overflow at probe size; raise --probe-chars"
        except Exception as e:
            msg = str(e)
            nums = re.findall(r"(\d{4,7})", msg)
            if nums:
                ctx = "%s tok" % max(int(x) for x in nums[-2:]) if len(nums) > 1 else nums[-1]
                ctx = nums[-1] + " tok"
            note = msg[:90].replace("\n", " ")

        # is max_tokens honoured?
        mt_ok = "?"
        try:
            out = await be.generate(
                [{"role": "user", "content":
                  "Count slowly from 1 to 400, one number per line."}],
                n=1, temperature=0.0, max_tokens=48, cache_nonce="mtprobe")
            got = len((out[0] or "").split())
            mt_ok = "YES (~%d w)" % got if got < 90 else "NO (%d w)" % got
        except Exception as e:
            mt_ok = "err: %s" % str(e)[:30]
        print("%-20s %14s %14s %s" % (m, ctx, mt_ok, note))
        rows.append({"model": m, "ctx": ctx, "max_tokens_honoured": mt_ok,
                     "note": note})
        be.close()
    return rows


def pick_cases(traces, n_wrong, n_right):
    wrong, right = [], []
    for t in traces:
        gold = t.get("gold", "")
        q = t.get("question", "")
        for msg in t.get("messages", []):
            if msg.get("role") != "solver" or int(msg.get("round", 0)) != 1:
                continue
            ans = msg.get("answer")
            if not ans:
                continue
            case = {"pid": t["pid"], "question": q,
                    "solution": msg.get("text", ""), "gold": gold}
            (right if is_correct(str(ans), str(gold)) else wrong).append(case)
            break
    return wrong[:n_wrong], right[:n_right]


async def critic_bakeoff(models, traces_path, n_wrong, n_right, cache):
    traces = load_any(traces_path)
    wrong, right = pick_cases(traces, n_wrong, n_right)
    print("cases: %d wrong, %d right\n" % (len(wrong), len(right)))
    if len(wrong) < 5:
        sys.exit("Not enough wrong round-1 solutions to measure recall.")

    async def flagged(be, case):
        p = CRITIC_PROMPT.format(question=case["question"],
                                 solution=case["solution"])
        try:
            out = await be.generate([{"role": "user", "content": p}], n=1,
                                    temperature=0.0, max_tokens=256,
                                    cache_nonce="bakeoff")
            head = (out[0] or "")[:200].upper()
        except Exception:
            return None
        if "VERDICT: ERROR" in head:
            return True
        if "VERDICT: CORRECT" in head:
            return False
        return None

    print("%-20s %8s %8s %8s %8s" % ("model", "recall", "fpr", "prec", "unparsed"))
    rows = []
    for m in models:
        be = mk(m, cache, max_tokens=256)
        rw = await asyncio.gather(*[flagged(be, c) for c in wrong])
        rr = await asyncio.gather(*[flagged(be, c) for c in right])
        be.close()
        tp = sum(1 for x in rw if x is True)
        fn = sum(1 for x in rw if x is False)
        fp = sum(1 for x in rr if x is True)
        tn = sum(1 for x in rr if x is False)
        bad = sum(1 for x in rw + rr if x is None)
        recall = tp / max(1, tp + fn)
        fpr = fp / max(1, fp + tn)
        prec = tp / max(1, tp + fp)
        print("%-20s %8.2f %8.2f %8.2f %8d" % (m, recall, fpr, prec, bad))
        rows.append({"model": m, "recall": recall, "fpr": fpr,
                     "precision": prec, "unparsed": bad,
                     "tp": tp, "fn": fn, "fp": fp, "tn": tn})

    print("\nBASELINE: deepseek-v3.2 critiquing its own output in the v1 traces")
    print("          scored recall 0.10, precision 0.54.")
    print("DECISION: pick the critic with the highest recall whose fpr stays")
    print("          below ~0.25. A critic that flags everything is useless:")
    print("          it gives the solver no signal about WHICH step to fix.")
    best = max(rows, key=lambda r: r["recall"] - r["fpr"]) if rows else None
    if best:
        print("\nbest by (recall - fpr): %s  recall %.2f  fpr %.2f"
              % (best["model"], best["recall"], best["fpr"]))
        if best["recall"] < 0.25:
            print("WARNING: even the best candidate is under 0.25 recall. The")
            print("problem is the PROTOCOL (stateless self-refinement), not the")
            print("model. Fix the harness before regenerating.")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["limits", "critic", "both"], default="both")
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--traces", default="data/traces.jsonl")
    ap.add_argument("--n-wrong", type=int, default=20)
    ap.add_argument("--n-right", type=int, default=20)
    ap.add_argument("--probe-chars", type=int, default=200000)
    ap.add_argument("--cache", default="/content/drive/MyDrive/cmd/cache_audit.jsonl")
    ap.add_argument("--out", default="results/provider_audit.json")
    args = ap.parse_args()

    out = {}
    if args.mode in ("limits", "both"):
        print("=== CONTEXT AND max_tokens ===")
        out["limits"] = asyncio.run(
            probe_limits(args.models, args.cache, args.probe_chars))
    if args.mode in ("critic", "both"):
        print("\n=== CRITIC BAKE-OFF ===")
        out["critic"] = asyncio.run(
            critic_bakeoff(args.models, args.traces, args.n_wrong,
                           args.n_right, args.cache))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w", encoding="utf-8"), indent=2)
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
