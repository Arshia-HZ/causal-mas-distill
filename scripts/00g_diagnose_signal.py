#!/usr/bin/env python3
"""00g_diagnose_signal.py  v2 -- ZERO API calls.

v2 fixes a grader-mismatch bug in v1: v1 recomputed correctness with a weak
built-in string matcher and then compared the result against probe pass rates
that had been graded by eval/grade.py. That understated debate accuracy by
several points. v2 uses eval.grade.is_correct whenever it can import it, and
refuses to draw cross-grader conclusions when it cannot.

Run from the repo root so that eval/ is importable:
  python scripts/00g_diagnose_signal.py --traces data/traces.jsonl \
      --max-tokens 1024 --probed data/probed_all.json
"""
import argparse
import json
import os
import re
import sys
from collections import Counter, defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for p in (ROOT, os.getcwd()):
    if p not in sys.path:
        sys.path.insert(0, p)

_grade = None
GRADER = "builtin"
try:
    from eval.grade import is_correct as _grade
    GRADER = "eval.grade.is_correct"
except Exception:
    pass

HAVE_MV = False
try:
    import math_verify  # noqa: F401
    HAVE_MV = True
except Exception:
    pass


def load_traces(path):
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1)
        f.seek(0)
        if first == "[":
            return json.load(f), "json-array"
        rows = []
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows, "jsonl"


def extract_boxed(text):
    i = text.rfind("\\boxed")
    if i < 0:
        return None, False
    j = text.find("{", i)
    if j < 0:
        return None, True
    depth = 0
    for k in range(j, len(text)):
        if text[k] == "{":
            depth += 1
        elif text[k] == "}":
            depth -= 1
            if depth == 0:
                return text[j + 1:k], False
    return None, True


def extract_answer(text):
    """Returns (answer_or_None, was_cut_mid_box)."""
    if not text:
        return None, False
    val, cut = extract_boxed(text)
    if val is not None:
        return val, cut
    hits = re.findall(r"####\s*(.+)", text)
    if hits:
        return hits[-1].strip(), cut
    hits = re.findall(r"(?:final answer|the answer)\s*(?:is|:)\s*(.+)", text, re.I)
    if hits:
        return hits[-1].strip(), cut
    return None, cut


def norm(a):
    if a is None:
        return None
    s = str(a).strip()
    for junk in ("$", " ", "\\left", "\\right", "\\!", "\\,", "\\;"):
        s = s.replace(junk, "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{([^}]*)\}", r"\1", s)
    for pre in ("\\dfrac", "\\tfrac"):
        if s.startswith(pre):
            s = "\\frac" + s[len(pre):]
    s = s.rstrip(".").rstrip("$")
    if s.endswith("%"):
        s = s[:-1]
    return s.lower()


def answers_agree(a, b):
    """String-level agreement between two model answers. No gold involved,
    so this is symmetric and grader-independent."""
    na, nb = norm(a), norm(b)
    if na is None or nb is None:
        return False
    if na == nb:
        return True
    try:
        return abs(float(na) - float(nb)) < 1e-6
    except Exception:
        return False


def correct(text, gold):
    """Grade a raw message against gold using the strongest grader available."""
    if text is None:
        return False
    if _grade is not None:
        try:
            return bool(_grade(text, gold))
        except Exception:
            pass
    a, _ = extract_answer(text)
    return answers_agree(a, gold)


NO_ERROR = (
    "no factual error", "no error", "is correct", "solution is correct",
    "no mistakes", "looks correct", "no issues", "correct as written",
    "i agree", "agree with the solution", "no corrections",
)


def critic_says_clean(text):
    t = (text or "").lower()
    return any(k in t for k in NO_ERROR)


def msgs(t):
    return t.get("messages") or []


def pct(a, b):
    return 0.0 if not b else 100.0 * a / b


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--probed", default=None)
    ap.add_argument("--dump-missing", type=int, default=0,
                    help="print N round-2/3 solver messages that lack an answer")
    ap.add_argument("--dump-critics", type=int, default=0,
                    help="print N critic messages that missed a real error")
    args = ap.parse_args()

    traces, fmt = load_traces(args.traces)
    cap_chars = int(args.max_tokens * 3.6)
    near = int(cap_chars * 0.92)

    bar = "=" * 66
    dash = "-" * 66
    print(bar)
    print("SIGNAL DIAGNOSIS v2  (%d traces, %s)" % (len(traces), fmt))
    print("grader                    : %s" % GRADER)
    print("math_verify available     : %s" % HAVE_MV)
    if _grade is None:
        print("  !! eval.grade NOT importable. Run from the repo root.")
        print("  !! Absolute accuracies below are NOT comparable to probe")
        print("  !! pass rates, which were graded by eval/grade.py.")
    print(bar)

    # ---- 1. missing answers -------------------------------------------
    miss_round = Counter()
    miss_near = miss_short = miss_cut = 0
    n_solver = 0
    samples = []
    for t in traces:
        for m in msgs(t):
            if m.get("role") != "solver":
                continue
            n_solver += 1
            a, cut = extract_answer(m.get("text") or "")
            if m.get("answer"):
                a = m["answer"]
            if a is not None:
                continue
            txt = m.get("text") or ""
            miss_round[m.get("round")] += 1
            if len(txt) >= near:
                miss_near += 1
            else:
                miss_short += 1
                if len(samples) < args.dump_missing:
                    samples.append((t.get("trace_id"), m.get("mid"), len(txt), txt))
            if cut:
                miss_cut += 1

    n_miss = sum(miss_round.values())
    print("SOLVER MESSAGES WITH NO EXTRACTABLE ANSWER")
    print("  total                   : %d / %d  (%.1f%%)"
          % (n_miss, n_solver, pct(n_miss, n_solver)))
    print("  by round                : %s" % dict(sorted(miss_round.items())))
    print("  AT the char cap         : %d  (%.1f%% of missing)"
          % (miss_near, pct(miss_near, n_miss)))
    print("  well SHORT of the cap   : %d  (%.1f%% of missing)"
          % (miss_short, pct(miss_short, n_miss)))
    print("  cut off mid-\\boxed{}    : %d" % miss_cut)
    if 1 not in miss_round and n_miss:
        print("  => ZERO in round 1. The solve prompt is fine; the REVISION")
        print("     prompt is what fails to demand a restated answer.")
    for tid, mid, ln, txt in samples:
        print("  " + dash)
        print("  %s %s  (%d chars)" % (tid, mid, ln))
        print("  ..." + txt[-600:].replace("\n", "\n  "))

    # ---- 2. does the debate move the answer ---------------------------
    print(dash)
    print("DOES THE DEBATE MOVE THE ANSWER?  (same grader on both sides)")
    cell = Counter()
    changed = n_eval = skipped = 0
    file_correct = 0
    n_file = 0
    for t in traces:
        gold = t.get("gold")
        ms = msgs(t)
        if t.get("final_correct") is not None:
            n_file += 1
            file_correct += 1 if t["final_correct"] else 0
        solvers = [m for m in ms if m.get("role") == "solver"]
        if not solvers:
            continue
        first = solvers[0]
        a1, _ = extract_answer(first.get("text") or "")
        if a1 is None:
            skipped += 1
            continue
        fin = t.get("final_answer")
        term = [m for m in ms if m.get("role") == "verifier"]
        fin_text = term[-1].get("text") if term else None
        if fin is None and fin_text:
            fin, _ = extract_answer(fin_text)
        n_eval += 1
        if not answers_agree(a1, fin):
            changed += 1
        c1 = correct(first.get("text"), gold)
        cf = t.get("final_correct")
        if cf is None:
            cf = correct(fin_text, gold)
        cell[(bool(c1), bool(cf))] += 1

    r1_acc = (cell[(True, True)] + cell[(True, False)]) / max(1, n_eval)
    fin_acc = (cell[(True, True)] + cell[(False, True)]) / max(1, n_eval)
    print("  traces scored            : %d  (skipped %d)" % (n_eval, skipped))
    print("  final != round-1 answer  : %d  (%.1f%%)" % (changed, pct(changed, n_eval)))
    print("  round-1 solver accuracy  : %.3f" % r1_acc)
    print("  final accuracy           : %.3f" % fin_acc)
    print("  delta from debating      : %+.3f   <- ROBUST, one grader" % (fin_acc - r1_acc))
    print("  rescued (wrong -> right) : %d" % cell[(False, True)])
    print("  broken  (right -> wrong) : %d" % cell[(True, False)])
    net = cell[(False, True)] - cell[(True, False)]
    print("  net                      : %+d traces" % net)
    if n_file:
        print("  file-recorded final_correct: %.3f  (%d traces, grade.py at"
              % (file_correct / n_file, n_file))
        print("                               generation time)")

    # ---- 3. critic quality --------------------------------------------
    print(dash)
    print("CRITIC QUALITY  (does it catch REAL errors?)")
    cm = Counter()
    by_round = defaultdict(Counter)
    crit_samples = []
    for t in traces:
        gold = t.get("gold")
        ms = msgs(t)
        pos = {m.get("mid"): m for m in ms}
        for m in ms:
            if m.get("role") != "critic":
                continue
            rnd = m.get("round")
            target = pos.get("r%s.solver" % rnd)
            if target is None:
                continue
            ta, _ = extract_answer(target.get("text") or "")
            if ta is None:
                continue
            wrong = not correct(target.get("text"), gold)
            clean = critic_says_clean(m.get("text") or "")
            cm[(wrong, clean)] += 1
            by_round[rnd]["clean" if clean else "flags"] += 1
            if wrong and clean and len(crit_samples) < args.dump_critics:
                crit_samples.append((t.get("trace_id"), m.get("mid"), gold,
                                     ta, m.get("text") or ""))

    tp, fn = cm[(True, False)], cm[(True, True)]
    fp, tn = cm[(False, False)], cm[(False, True)]
    tot = tp + fn + fp + tn
    print("  judgements scored        : %d" % tot)
    print("  solver WRONG, flagged    : %d" % tp)
    print("  solver WRONG, said fine  : %d   <- missed errors" % fn)
    print("  solver RIGHT, flagged    : %d   <- false alarms" % fp)
    print("  solver RIGHT, said fine  : %d" % tn)
    if tp + fn:
        print("  RECALL on real errors    : %.3f" % (tp / (tp + fn)))
    if tp + fp:
        print("  PRECISION when flagging  : %.3f" % (tp / (tp + fp)))
    print("  flag rate                : %.1f%%" % pct(tp + fp, tot))
    print("  verdicts by round        : %s"
          % {k: dict(v) for k, v in sorted(by_round.items())})
    if _grade is None:
        print("  !! recall is biased LOW without grade.py: correct answers")
        print("  !! misgraded as wrong land in the 'missed errors' cell.")
    for tid, mid, gold, ta, txt in crit_samples:
        print("  " + dash)
        print("  %s %s   gold=%s solver_said=%s" % (tid, mid, gold, ta))
        print("  " + txt[:700].replace("\n", "\n  "))

    # ---- 4. gate preview ---------------------------------------------
    if args.probed:
        print(dash)
        print("FREE GATE PREVIEW (probe data, no API calls)")
        try:
            with open(args.probed, "r", encoding="utf-8") as f:
                probed = json.load(f)
            rate = {}
            for p in probed:
                pid = p.get("pid") or p.get("id")
                if pid is not None and p.get("pass_rate") is not None:
                    rate[pid] = float(p["pass_rate"])
            pids = sorted({t.get("pid") for t in traces})
            ps = [rate[p] for p in pids if p in rate]
            if ps:
                mean_p = sum(ps) / len(ps)
                at3 = sum(1.0 - (1.0 - p) ** 3 for p in ps) / len(ps)
                at6 = sum(1.0 - (1.0 - p) ** 6 for p in ps) / len(ps)
                dbg = (file_correct / n_file) if n_file else fin_acc
                lo = min(ps)
                hi = max(ps)
                print("  problems matched         : %d / %d" % (len(ps), len(pids)))
                print("  probe pass rate range    : %.3f .. %.3f" % (lo, hi))
                print("  arm A, single sample     : %.3f" % mean_p)
                print("  pass@3 UPPER bound       : %.3f" % at3)
                print("  pass@6 UPPER bound       : %.3f   <- SC@6 ceiling" % at6)
                print("  debate, grade.py         : %.3f" % dbg)
                print("  debate - single          : %+.3f" % (dbg - mean_p))
                if mean_p > 0.65:
                    print("  !! mean pass rate %.3f means this 'headroom' set is" % mean_p)
                    print("  !! mostly near-ceiling problems. A band of 0<p<1 at")
                    print("  !! k=32 admits p=31/32. Re-band before concluding.")
                if dbg > at6:
                    print("  => debate beats the pass@6 ceiling. Strong. Run 00c.")
                elif dbg > mean_p:
                    print("  => debate beats 1 sample but sits far below the")
                    print("     pass@6 ceiling. 00c must arbitrate vs SC@6.")
                else:
                    print("  => debate at or below a single sample.")
        except Exception as e:
            print("  could not read --probed: %s: %s" % (type(e).__name__, e))

    print(bar)
    return 0


if __name__ == "__main__":
    sys.exit(main())
