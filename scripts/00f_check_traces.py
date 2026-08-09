#!/usr/bin/env python3
"""Offline validation of generated debate traces. Zero API calls.

Run this BEFORE the headroom audit. The audit costs money and hours; this
costs a second and catches the failure modes that silently corrupt arm C:

  * truncation      a message that hit the output cap loses its \\boxed{},
                    grades as wrong, and makes the debate look worse than it
                    is. At max_tokens=768 on MATH this is a real risk.
  * short traces    a dropped critique breaks the 6-generation budget match,
                    so --n-sc 6 stops being the right baseline.
  * missing pids    problems that failed generation are silently absent.
  * dead critics    if every critic says 'no errors', there is no
                    error-correction signal to distil and the whole thesis
                    premise is unsupported on this data.

Usage:
  python scripts/00f_check_traces.py --traces data/traces.jsonl \\
      --targets data/trace_targets.json --max-tokens 768
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.grade import extract_answer


def load_traces(path):
    raw = open(path).read()
    if raw.lstrip().startswith("["):
        return json.loads(raw), "json-array"
    return [json.loads(ln) for ln in raw.splitlines() if ln.strip()], "jsonl"


NO_ERROR_MARKERS = (
    "no factual errors",
    "no errors",
    "is correct",
    "solution is correct",
    "no mistakes",
    "looks correct",
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--targets", default=None,
                    help="trace_targets.json, to detect problems that failed.")
    ap.add_argument("--max-tokens", type=int, default=768,
                    help="Output cap used during generation.")
    ap.add_argument("--expect-messages", type=int, default=6,
                    help="Expected messages per trace for max_rounds=3.")
    args = ap.parse_args()

    traces, fmt = load_traces(args.traces)
    bar = "=" * 66
    print(bar)
    print("TRACE CHECK")
    print(bar)
    print("file format               : " + fmt)
    if fmt == "json-array":
        print("  NOTE: written as a JSON array despite the .jsonl name.")
        print("  00c and 02 now sniff for this, so no action needed.")
    print("traces                    : " + str(len(traces)))

    if not traces:
        print("no traces. stop.")
        return 2

    per_pid = defaultdict(int)
    for t in traces:
        per_pid[t.get("pid")] += 1
    print("unique problems           : " + str(len(per_pid)))
    print("seeds per problem         : " + str(dict(Counter(per_pid.values()))))

    if args.targets:
        want = set()
        for p in json.load(open(args.targets)):
            want.add(p.get("pid"))
        missing = want - set(per_pid)
        print("targets                   : " + str(len(want)))
        if missing:
            print("  MISSING " + str(len(missing)) + " problems produced no trace:")
            print("  " + ", ".join(sorted(missing)[:10]))
        else:
            print("  all targets produced traces")

    # Truncation. The cap is in tokens; we only have characters. ~3.6 chars per
    # token is a reasonable rate for English maths prose, so anything within 8%
    # of the implied character ceiling is suspicious. The decisive signal is a
    # solver or verifier message with no extractable answer.
    char_cap = int(args.max_tokens * 3.6)
    near_cap = 0
    no_answer = defaultdict(int)
    role_counts = Counter()
    msg_counts = Counter()
    short_traces = []
    dead_critics = 0
    total_critics = 0

    for t in traces:
        msgs = t.get("messages", [])
        msg_counts[len(msgs)] += 1
        if len(msgs) < args.expect_messages:
            short_traces.append(t.get("trace_id"))
        for m in msgs:
            role = m.get("role", "?")
            role_counts[role] += 1
            text = m.get("text") or ""
            if len(text) >= char_cap * 0.92:
                near_cap += 1
            if role in ("solver", "verifier"):
                if not (m.get("answer") or extract_answer(text)):
                    no_answer[role] += 1
            if role == "critic":
                total_critics += 1
                low = text.lower()
                if any(k in low for k in NO_ERROR_MARKERS):
                    dead_critics += 1

    print("-" * 66)
    print("messages per trace        : " + str(dict(sorted(msg_counts.items()))))
    print("roles                     : " + str(dict(role_counts)))
    if short_traces:
        print("SHORT TRACES              : " + str(len(short_traces))
              + "  <- budget match with --n-sc 6 is broken for these")
        print("  e.g. " + ", ".join(short_traces[:5]))

    n_msgs = sum(role_counts.values())
    print("-" * 66)
    print("TRUNCATION (cap=" + str(args.max_tokens) + " tokens)")
    pct = 100.0 * near_cap / max(n_msgs, 1)
    print("  messages near char cap  : %d / %d  (%.1f%%)" % (near_cap, n_msgs, pct))
    for role in ("solver", "verifier"):
        tot = role_counts.get(role, 0)
        bad = no_answer.get(role, 0)
        if tot:
            print("  %-8s with NO answer : %d / %d  (%.1f%%)"
                  % (role, bad, tot, 100.0 * bad / tot))

    print("-" * 66)
    if total_critics:
        dp = 100.0 * dead_critics / total_critics
        print("critics finding no error  : %d / %d  (%.1f%%)"
              % (dead_critics, total_critics, dp))
        if dp > 80.0:
            print("  WARNING: the critic almost never disagrees. There is little")
            print("  error-correction signal here to attribute utility to.")

    # ---- seed independence -------------------------------------------------
    # The decisive test. Sibling seeds are supposed to be independent samples;
    # if the round-1 texts are byte-identical they are one sample counted N
    # times, every CI computed over them is too narrow, and per-problem scores
    # collapse to 0/N or N/N.
    print("-" * 66)
    by_pid_traces = defaultdict(list)
    for t in traces:
        by_pid_traces[t.get("pid")].append(t)

    def first_solver(t):
        for m in t.get("messages", []):
            if m.get("role") == "solver":
                return m.get("text") or ""
        return ""

    multi = [v for v in by_pid_traces.values() if len(v) > 1]
    if multi:
        collapsed_r1 = 0
        collapsed_all = 0
        for group in multi:
            r1 = set(first_solver(t) for t in group)
            if len(r1) == 1:
                collapsed_r1 += 1
            whole = set(
                "\x00".join((m.get("text") or "") for m in t.get("messages", []))
                for t in group
            )
            if len(whole) == 1:
                collapsed_all += 1
        pr1 = 100.0 * collapsed_r1 / len(multi)
        print("SEED INDEPENDENCE over %d multi-seed problems" % len(multi))
        print("  identical round-1 solver  : %d  (%.1f%%)" % (collapsed_r1, pr1))
        print("  identical ENTIRE trace    : %d  (%.1f%%)" % (collapsed_all,
              100.0 * collapsed_all / len(multi)))
        if pr1 > 10.0:
            print("  FATAL: seeds are duplicates, not samples. Every 'n=3 seeds'")
            print("  claim is really n=1, and confidence intervals built on them")
            print("  are fiction. Cause: identical payloads share a cache key.")
    else:
        print("SEED INDEPENDENCE       : only one trace per problem, nothing to check")

    correct = [1.0 if t.get("final_correct") else 0.0 for t in traces]
    print("-" * 66)
    print("debate accuracy (traces)  : %.3f" % (sum(correct) / len(correct)))
    by_pid = defaultdict(list)
    for t in traces:
        by_pid[t.get("pid")].append(1.0 if t.get("final_correct") else 0.0)
    means = [sum(v) / len(v) for v in by_pid.values()]
    print("debate accuracy (per-pid) : %.3f" % (sum(means) / len(means)))
    spread = Counter(round(m, 3) for m in means)
    print("per-pid spread            : " + str(dict(sorted(spread.items()))))

    if len(spread) <= 2 and len(means) > 30 and all(
        m in (0.0, 1.0) for m in spread
    ):
        print("  ^ every problem is all-or-nothing across seeds. Under genuine")
        print("    independence this is essentially impossible; it is the")
        print("    fingerprint of collapsed seeds.")

    print(bar)
    fatal = False
    if pct > 5.0 or no_answer.get("verifier", 0) > 0.05 * role_counts.get("verifier", 1):
        print("VERDICT: truncation is material. Regenerate with a higher cap")
        print("  before auditing, or the debate arm is unfairly penalised.")
        fatal = True
    if short_traces:
        print("VERDICT: some traces are short. Either drop them or account for")
        print("  their smaller budget explicitly.")
        fatal = True
    if not fatal:
        print("VERDICT: traces look clean. Proceed to the audit.")
    print(bar)
    return 1 if fatal else 0


if __name__ == "__main__":
    sys.exit(main())
