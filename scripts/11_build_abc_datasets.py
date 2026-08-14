#!/usr/bin/env python3
"""
Build MATCHED Arm A / Arm B / Arm C SFT datasets.

  Arm A = rejection-sampled correct solution        (STaR / RFT baseline)
  Arm B = full debate transcript                    (process hypothesis)
  Arm C = the debate's FINAL solution only          (product hypothesis)

WHY ARM C IS NEW AND WHY IT MATTERS
-----------------------------------
A vs B confounds two different claims:
  (i)  deliberation produces a BETTER SOLUTION  -> tested by C vs A
  (ii) the deliberation TEXT ITSELF is teachable -> tested by B vs C
Without C you cannot tell a null apart from two effects cancelling, and
"debate transcripts do not help" is the obvious, unpublishable result. The
A/C/B decomposition is the defensible contribution.
Arm C is also naturally length-matched to Arm A, so it is immune to the
context-window problem that cripples Arm B on a 4096-token student.

WHAT WENT WRONG ON 2026-08-13
-----------------------------
The token-matched build produced A over 67 problems and B over 22 problems.
Equal tokens, wildly unequal COVERAGE, because the length filter removed
whole problems from B only. That alone invalidates the comparison.
Fix: filter for length FIRST, then intersect the surviving problem sets
across all arms, and make the PRIMARY dataset one example per problem.

VARIANTS EMITTED
----------------
  *_eqp.jsonl  PRIMARY. One example per shared problem. Equal problems,
               equal examples, equal gradient updates. Only content differs.
               Token count differs between arms -- that is an inherent
               property of the treatment, not a removable confound.
  *_tok.jsonl  SECONDARY robustness check. Equal completion tokens.

USAGE
-----
  python scripts/11_build_abc_datasets.py \\
      --arm-a-pool data/arm_a_pool.jsonl \\
      --traces data/traces.jsonl \\
      --probed data/probed_all.json \\
      --clip-critic-words 200 --max-completion-tokens 3584 \\
      --eval-extra 200 --outdir data
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from pathlib import Path

STUDENT = "Qwen/Qwen2.5-1.5B-Instruct"
ARMS = ("a", "b", "c")


def load_jsonl(path):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_any(path):
    text = open(path, encoding="utf-8").read().strip()
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(l) for l in text.splitlines() if l.strip()]


def pick(d, *keys, default=""):
    for k in keys:
        v = d.get(k)
        if isinstance(v, str) and v.strip():
            return v
    return default


def is_eval_pid(pid, mod=5):
    return int(hashlib.md5(pid.encode()).hexdigest(), 16) % mod == 0


def clip_words(text, n):
    if n <= 0:
        return text
    w = text.split()
    return text if len(w) <= n else " ".join(w[:n]) + " [...]"


def render_transcript(trace, clip_critic=0, max_round=0):
    """Arm B. Labelled turns in order. Gold is never inserted (trap T6)."""
    msgs = list(trace.get("messages", []))
    if max_round > 0:
        keep = [m for m in msgs if int(m.get("round", 1)) <= max_round]
        if msgs and msgs[-1] not in keep:
            keep.append(msgs[-1])
        msgs = keep
    parts = []
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


# --- answer agreement helpers (for Arm C turn selection) ------------------
import re as _re

try:
    from eval.grade import is_correct as _grade_correct
except Exception:
    _grade_correct = None


def _norm_ans(s):
    s = (s or "").strip()
    for a in (r"\!", r"\,", r"\;", r"\ ", r"\left", r"\right"):
        s = s.replace(a, "")
    s = s.replace("$", "").replace(" ", "").replace("\n", "")
    s = s.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    s = _re.sub(r"(?<=\d),(?=\d\d\d)", "", s)
    return s.rstrip(".$%")


def agree_ans(a, b):
    """Cheap answer agreement. Strong enough here because the trace's final
    answer was already verified against gold (final_correct gate upstream);
    we only need to know which solver turn CARRIES that answer."""
    if _grade_correct is not None:
        try:
            return bool(_grade_correct(a, b))
        except Exception:
            pass
    a, b = _norm_ans(a), _norm_ans(b)
    if not a or not b:
        return False
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-6
    except Exception:
        return False


def render_final_solution(trace):
    """
    Arm C. The debate's final solution as a COMPLETE derivation, with every
    critique stripped out.

    v2 fix (2026-08-14): "the last solver turn with an answer" is wrong under
    the rcr_v3 prompts. When the critic agrees (~94% of critiques), the last
    solver turn is a short confirmation ("The final answer remains
    \\boxed{X}") -- measured mean 138 tokens, 0.15x of Arm A, with no
    reasoning to distill. Pick instead the LONGEST solver turn whose answer
    agrees with the trace's final answer: when the critic agreed, that is
    the round-1 full derivation; when a dispute led to a fix, it is the
    correcting revision. Returns "" when no solver turn carries the final
    answer (verifier-only rescues, ~2%), so that problem drops out of C --
    and therefore out of the shared set, keeping T11 coverage matching
    intact rather than training on a wrong answer.
    """
    final = (trace.get("final_answer") or "").strip()
    best = ""
    for m in trace.get("messages", []):
        if m.get("role") != "solver":
            continue
        t = (m.get("text") or "").strip()
        if not t:
            continue
        ans = (m.get("answer") or "").strip()
        if not ans:
            continue
        if final and not agree_ans(ans, final):
            continue
        if len(t) > len(best):
            best = t
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-a-pool", required=True)
    ap.add_argument("--traces", required=True)
    ap.add_argument("--probed", default=None)
    ap.add_argument("--budget-tokens", type=int, default=400000)
    ap.add_argument("--eval-mod", type=int, default=5)
    ap.add_argument("--eval-extra", type=int, default=0,
                    help="Add up to N extra eval problems drawn from --probed "
                         "that appear in NO training set. 32 eval problems "
                         "gives a +-9pt CI, which cannot resolve anything.")
    ap.add_argument("--clip-critic-words", type=int, default=200)
    ap.add_argument("--max-rounds-render", type=int, default=0,
                    help="Render only rounds <= R for Arm B (final message "
                         "always kept). Use 2 if Arm B still will not fit.")
    ap.add_argument("--max-completion-tokens", type=int, default=3584)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="data")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(STUDENT)

    def ntok(t):
        return len(tok(t, add_special_tokens=False)["input_ids"])

    def render_prompt(q):
        return tok.apply_chat_template(
            [{"role": "user", "content": q}],
            tokenize=False, add_generation_prompt=True)

    pool = {r["pid"]: r for r in load_jsonl(args.arm_a_pool)}
    traces = load_any(args.traces)

    qmap, gmap = {}, {}
    cand = {a: {} for a in ARMS}

    for pid, r in pool.items():
        sols = [s for s in r.get("solutions", []) if s and s.strip()]
        if sols:
            cand["a"][pid] = sols
        qmap[pid] = pick(r, "question", "problem")
        gmap[pid] = pick(r, "gold", "answer")

    for t in traces:
        if not t.get("final_correct"):
            continue
        pid = t["pid"]
        qmap.setdefault(pid, pick(t, "question", "problem"))
        gmap.setdefault(pid, pick(t, "gold", "answer"))
        b = render_transcript(t, args.clip_critic_words, args.max_rounds_render)
        if b.strip():
            cand["b"].setdefault(pid, []).append(b)
        c = render_final_solution(t)
        if c.strip():
            cand["c"].setdefault(pid, []).append(c)

    # ---- LENGTH FILTER FIRST, THEN INTERSECT (the 2026-08-13 fix) ---------
    print("raw problem coverage : " + "  ".join(
        "%s=%d" % (a.upper(), len(cand[a])) for a in ARMS))

    kept = {a: {} for a in ARMS}
    dropped = {a: 0 for a in ARMS}
    for a in ARMS:
        for pid, opts in cand[a].items():
            ok = []
            for o in opts:
                n = ntok(o)
                if n <= args.max_completion_tokens:
                    ok.append((o, n))
                else:
                    dropped[a] += 1
            if ok:
                kept[a][pid] = ok
    print("dropped over %d tok  : " % args.max_completion_tokens + "  ".join(
        "%s=%d" % (a.upper(), dropped[a]) for a in ARMS))
    print("after length filter  : " + "  ".join(
        "%s=%d" % (a.upper(), len(kept[a])) for a in ARMS))

    shared = sorted(set(kept["a"]) & set(kept["b"]) & set(kept["c"]))
    print("SHARED PROBLEMS (T3) : %d" % len(shared))
    if len(shared) < 20:
        print("\nWARNING: fewer than 20 shared problems. No estimator can "
              "rescue this. Generate debate traces for more problems, or "
              "lower --clip-critic-words / set --max-rounds-render 2.")
    if not shared:
        sys.exit("No shared problems.")

    train_pids = [p for p in shared if not is_eval_pid(p, args.eval_mod)]
    eval_pids = [p for p in shared if is_eval_pid(p, args.eval_mod)]
    print("train problems: %d   eval problems (shared): %d"
          % (len(train_pids), len(eval_pids)))

    rng = random.Random(args.seed)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    def dump(rows, name):
        with open(outdir / name, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def row(pid, completion, n):
        return {"pid": pid, "prompt": render_prompt(qmap.get(pid, "")),
                "completion": completion, "completion_tokens": n}

    # ---- PRIMARY: one example per shared problem --------------------------
    eqp = {}
    for a in ARMS:
        r = random.Random(args.seed)
        rows = []
        for pid in train_pids:
            o, n = r.choice(kept[a][pid])
            rows.append(row(pid, o, n))
        eqp[a] = rows
        dump(rows, "sft_arm_%s_eqp.jsonl" % a)

    # ---- SECONDARY: token-matched, still inside the shared pid set --------
    def build_tok(a, budget):
        order = list(train_pids)
        rng.shuffle(order)
        cur = {p: 0 for p in order}
        rows, used, progress = [], 0, True
        while progress and used < budget:
            progress = False
            for pid in order:
                i = cur[pid]
                opts = kept[a][pid]
                if i >= len(opts):
                    continue
                o, n = opts[i]
                cur[pid] = i + 1
                if used + n > budget:
                    continue
                rows.append(row(pid, o, n))
                used += n
                progress = True
                if used >= budget:
                    break
        return rows, used

    caps = {}
    for a in ARMS:
        caps[a] = sum(n for opts in (kept[a][p] for p in train_pids)
                      for _, n in opts)
    budget = min(args.budget_tokens, min(caps.values()))
    tokd = {}
    for a in ARMS:
        rows, used = build_tok(a, budget)
        tokd[a] = (rows, used)
        dump(rows, "sft_arm_%s_tok.jsonl" % a)

    # ---- eval set ---------------------------------------------------------
    ev = [{"pid": p, "question": qmap.get(p, ""), "gold": gmap.get(p, ""),
           "in_shared": True} for p in eval_pids]
    if args.eval_extra > 0 and args.probed:
        used_pids = set(train_pids) | {e["pid"] for e in ev}
        extra = []
        for p in load_any(args.probed):
            pid = p.get("pid")
            if not pid or pid in used_pids:
                continue
            q = pick(p, "question", "problem")
            g = pick(p, "gold", "answer")
            if q and g:
                extra.append({"pid": pid, "question": q, "gold": g,
                              "in_shared": False})
        random.Random(1234).shuffle(extra)
        ev += extra[:args.eval_extra]
    json.dump(ev, open(outdir / "eval_problems.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)

    # ---- report -----------------------------------------------------------
    print("\nPRIMARY  sft_arm_{a,b,c}_eqp.jsonl   (equal problems + examples)")
    print("%-4s %6s %12s %10s %9s" % ("arm", "n", "comp_tokens", "mean", "probs"))
    for a in ARMS:
        rows = eqp[a]
        tt = sum(r["completion_tokens"] for r in rows)
        print("%-4s %6d %12d %10.1f %9d"
              % (a.upper(), len(rows), tt, tt / max(1, len(rows)),
                 len({r["pid"] for r in rows})))
    ta = sum(r["completion_tokens"] for r in eqp["a"]) or 1
    for a in ("b", "c"):
        tt = sum(r["completion_tokens"] for r in eqp[a])
        print("  token ratio %s/A = %.2fx  (inherent, report it; do not "
              "'correct' it)" % (a.upper(), tt / ta))

    print("\nSECONDARY sft_arm_{a,b,c}_tok.jsonl  (equal completion tokens)")
    print("%-4s %6s %12s %10s %9s" % ("arm", "n", "comp_tokens", "mean", "probs"))
    covs = []
    for a in ARMS:
        rows, used = tokd[a]
        cov = len({r["pid"] for r in rows})
        covs.append(cov)
        print("%-4s %6d %12d %10.1f %9d"
              % (a.upper(), len(rows), used, used / max(1, len(rows)), cov))
    toks = [t[1] for t in tokd.values()]
    if max(toks) > 0 and (max(toks) - min(toks)) / max(toks) > 0.01:
        print("FAIL: arms differ by more than 1% of completion tokens.")
        sys.exit(3)
    if max(covs) > 0 and min(covs) / max(covs) < 0.75:
        print(f"FAIL: problem coverage differs by more than 25% across arms "
              f"({covs}). This is exactly the 2026-08-13 failure. Use the _eqp "
              f"datasets as primary.")

    print("\neval problems -> %s (%d shared + %d extra)"
          % (outdir / "eval_problems.json", len(eval_pids),
             len(ev) - len(eval_pids)))
    if len(ev) < 150:
        print("WARNING: %d eval problems gives roughly a +-%.0f point 95%% CI. "
              "Pass --eval-extra 200." % (len(ev), 100 * 1.96 * 0.5 / (len(ev) ** 0.5)))

    if args.probed:
        pr = {p.get("pid"): float(p.get("pass_rate", 0))
              for p in load_any(args.probed)}
        ceil = sum(1 for p in train_pids if pr.get(p, 0) >= 1.0)
        print("\nDIFFICULTY (T7): %d/%d train problems at teacher ceiling"
              % (ceil, len(train_pids)))


if __name__ == "__main__":
    main()
