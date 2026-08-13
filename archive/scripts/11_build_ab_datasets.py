#!/usr/bin/env python3
"""
Build the matched Arm A / Arm B SFT datasets. This is the scientific core.

Arm A = SFT on rejection-sampled correct solutions   (STaR / RFT baseline)
Arm B = SFT on full debate transcripts               (the hypothesis)

The comparison is only meaningful if four things are matched:
  T3  identical problem sets  -> intersect the arms
  T1  real tokenizer counts   -> never len(text)//4
  T2  completion tokens only  -> loss is computed on the completion
  T8  held-out eval           -> no eval pid appears in either training set

And one thing cannot be matched, so it is reported twice:
  T4  a transcript is ~6x longer than a solution. At equal tokens Arm A gets
      ~6x more examples. That is the honest primary comparison, but it
      confounds content with number of gradient updates. So this script also
      emits a matched-EXAMPLE-COUNT variant. Run both. If they disagree,
      report both.

USAGE
-----
  python scripts/11_build_ab_datasets.py \\
      --arm-a-pool data/arm_a_pool.jsonl \\
      --traces data/traces.jsonl \\
      --probed data/probed_all.json \\
      --budget-tokens 400000 \\
      --outdir data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

STUDENT = "Qwen/Qwen2.5-1.5B-Instruct"


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_traces(path):
    """traces.jsonl may be a JSON array or line-delimited. Accept both."""
    text = open(path, encoding="utf-8").read().strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def is_eval_pid(pid: str, mod: int = 5) -> bool:
    """Deterministic 80/20 split. Same pid always lands on the same side."""
    return int(hashlib.md5(pid.encode()).hexdigest(), 16) % mod == 0


def clip_words(text: str, n: int) -> str:
    """Keep the first n words. Used to emulate a length-capped critic."""
    if n <= 0:
        return text
    w = text.split()
    if len(w) <= n:
        return text
    return " ".join(w[:n]) + " [...]"


def render_transcript(trace: dict, clip_critic: int = 0) -> str:
    """
    Arm B completion. Labelled turns, in order. The gold answer is never
    inserted (trap T6): the target is the teacher's own text.

    LENGTH PROBLEM (measured on the 314 correct v1 traces):
      median transcript = 25,213 chars ~= 8,400 student tokens
      critic messages are 49.3% of all characters, as much as all three
      solvers combined
      92.7% of transcripts exceed the 4096-token max_seq_length in
      configs/student_qwen2.5_1.5b.yaml

    Truncation cuts the END of the sequence, which is exactly where the final
    answer lives. Training Arm B on truncated transcripts teaches the student
    to ramble and never conclude, and would produce "B loses" as a pure
    artefact of sequence length. Either regenerate with a length-capped critic
    (prompts.py v3 caps at 200 words) or pass --clip-critic-words to emulate
    that cap post hoc for a free first read.
    """
    parts = []
    msgs = trace.get("messages", [])
    for i, m in enumerate(msgs):
        text = (m.get("text") or "").strip()
        if not text:
            continue
        role = m.get("role", "agent")
        if role == "critic" and clip_critic > 0:
            text = clip_words(text, clip_critic)
        label = "final answer" if i == len(msgs) - 1 else role
        parts.append("[%s]\n%s" % (label, text))
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-a-pool", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--probed", default=None,
                    help="probed_all.json, used only to tag difficulty")
    ap.add_argument("--budget-tokens", type=int, default=400000,
                    help="completion-token budget PER ARM")
    ap.add_argument("--eval-mod", type=int, default=5)
    ap.add_argument("--clip-critic-words", type=int, default=0,
                    help="Clip each critic message to N words when rendering "
                         "Arm B. 200 emulates the v3 length-capped critic and "
                         "brings the median transcript under the student's "
                         "4096-token context. 0 = no clipping.")
    ap.add_argument("--max-completion-tokens", type=int, default=3584,
                    help="Drop any candidate whose completion exceeds this. "
                         "Must leave room for the prompt inside "
                         "max_seq_length. Silent truncation is worse than "
                         "dropping the example.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(STUDENT)

    def ntok(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    def render_prompt(question: str) -> str:
        return tok.apply_chat_template(
            [{"role": "user", "content": question}],
            tokenize=False, add_generation_prompt=True,
        )

    pool = {r["pid"]: r for r in load_jsonl(args.arm_a_pool)}
    traces = load_traces(args.traces)

    # ---- candidate examples per problem -----------------------------------
    cand_a: dict[str, list[str]] = {}
    for pid, r in pool.items():
        sols = [s for s in r.get("solutions", []) if s and s.strip()]
        if sols:
            cand_a[pid] = sols

    cand_b: dict[str, list[str]] = {}
    qmap: dict[str, str] = {}
    gmap: dict[str, str] = {}
    for t in traces:
        if not t.get("final_correct"):
            continue
        body = render_transcript(t, clip_critic=args.clip_critic_words)
        if not body.strip():
            continue
        pid = t["pid"]
        cand_b.setdefault(pid, []).append(body)
        qmap[pid] = t.get("question", "")
        gmap[pid] = t.get("gold", "")

    for pid, r in pool.items():
        qmap.setdefault(pid, r.get("question", ""))
        gmap.setdefault(pid, r.get("gold", ""))

    pids_a, pids_b = set(cand_a), set(cand_b)
    shared = sorted(pids_a & pids_b)
    print("problems with a correct RS solution : %d" % len(pids_a))
    print("problems with a correct debate trace: %d" % len(pids_b))
    print("INTERSECTION (trap T3)              : %d" % len(shared))
    if not shared:
        sys.exit("No shared problems. Check pid formats match across files.")

    eval_pids = [p for p in shared if is_eval_pid(p, args.eval_mod)]
    train_pids = [p for p in shared if not is_eval_pid(p, args.eval_mod)]
    print("train problems: %d   eval problems: %d" % (len(train_pids), len(eval_pids)))

    # ---- round-robin fill to the token budget -----------------------------
    rng = random.Random(args.seed)

    dropped_long = [0]

    def build(cands: dict[str, list[str]], budget: int):
        order = list(train_pids)
        rng.shuffle(order)
        cursor = {p: 0 for p in order}
        rows, used = [], 0
        progress = True
        while progress and used < budget:
            progress = False
            for pid in order:
                i = cursor[pid]
                opts = cands.get(pid, [])
                if i >= len(opts):
                    continue
                completion = opts[i]
                cursor[pid] = i + 1
                c = ntok(completion)
                if c > args.max_completion_tokens:
                    dropped_long[0] += 1
                    continue
                if used + c > budget:
                    continue
                rows.append({
                    "pid": pid,
                    "prompt": render_prompt(qmap[pid]),
                    "completion": completion,
                    "completion_tokens": c,
                })
                used += c
                progress = True
                if used >= budget:
                    break
        return rows, used

    dropped_long[0] = 0
    rows_a, tok_a = build(cand_a, args.budget_tokens)
    dropped_a = dropped_long[0]
    dropped_long[0] = 0
    rows_b, tok_b = build(cand_b, args.budget_tokens)
    dropped_b = dropped_long[0]
    print("dropped for exceeding %d completion tokens: A=%d  B=%d"
          % (args.max_completion_tokens, dropped_a, dropped_b))
    if dropped_b > 0 and args.clip_critic_words == 0:
        print("HINT: pass --clip-critic-words 200. Critic text is ~49%% of "
              "transcript characters and is what pushes Arm B over the limit.")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def dump(rows, name):
        p = outdir / name
        with open(p, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        return p

    dump(rows_a, "sft_arm_a.jsonl")
    dump(rows_b, "sft_arm_b.jsonl")

    # ---- matched-example-count variant (trap T4) --------------------------
    n_eq = min(len(rows_a), len(rows_b))
    dump(rows_a[:n_eq], "sft_arm_a_eqn.jsonl")
    dump(rows_b[:n_eq], "sft_arm_b_eqn.jsonl")

    json.dump(
        [{"pid": p, "question": qmap[p], "gold": gmap[p]} for p in eval_pids],
        open(outdir / "eval_problems.json", "w", encoding="utf-8"),
        indent=2, ensure_ascii=False,
    )

    # ---- report ------------------------------------------------------------
    def summarise(rows, toks, label):
        n = len(rows)
        cov = len({r["pid"] for r in rows})
        mean = toks / n if n else 0
        print("%-8s %6d %12d %10.1f %9d" % (label, n, toks, mean, cov))

    print("\nMATCHED-TOKEN DATASETS")
    print("%-8s %6s %12s %10s %9s" % ("arm", "n", "comp_tokens", "mean_tok", "problems"))
    summarise(rows_a, tok_a, "A (RS)")
    summarise(rows_b, tok_b, "B (debate)")

    if max(tok_a, tok_b) > 0:
        gap = abs(tok_a - tok_b) / max(tok_a, tok_b)
        print("\ntoken gap: %.2f%%" % (100 * gap))
        if gap > 0.01:
            print("FAIL: arms differ by more than 1%% of completion tokens.")
            print("Lower --budget-tokens until both arms saturate, or the arm")
            print("with fewer candidates has run out of data. Check coverage.")
            sys.exit(3)

    print("\nmatched-example variant: %d examples per arm" % n_eq)
    print("eval problems -> %s (%d)" % (outdir / "eval_problems.json", len(eval_pids)))

    if args.probed:
        probed = {p.get("pid"): p for p in json.load(open(args.probed, encoding="utf-8"))}
        ceil = sum(1 for p in train_pids
                   if float(probed.get(p, {}).get("pass_rate", 0)) >= 1.0)
        print("\nDIFFICULTY (trap T7): %d/%d training problems are at teacher "
              "ceiling (p=1.0)" % (ceil, len(train_pids)))
        print("Report evaluation split by this. The pooled number hides the "
              "only interesting stratum.")


if __name__ == "__main__":
    main()
