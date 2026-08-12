#!/usr/bin/env python3
# 00i_coverage_gate.py -- ZERO API calls.
#
# The right question for a DISTILLATION thesis is not 'does debate beat
# self-consistency at inference'. It is 'does debate produce correct training
# traces for problems that plain rejection sampling, at the SAME generation
# budget, cannot reach'. That is the STaR baseline, and it is the only
# baseline a data-generation method actually has to beat.
#
#   python scripts/00i_coverage_gate.py --traces data/traces.jsonl \
#          --probed data/probed_all.json --gens-per-trace 6

import argparse
#!/usr/bin/env python3
import json, re
from collections import Counter, defaultdict

def load_traces(path):
    with open(path, "r", encoding="utf-8") as f:
        first = f.read(1); f.seek(0)
        if first == "[":
            return json.load(f)
        return [json.loads(l) for l in f if l.strip()]

def boxed(t):
    i = t.rfind("\\boxed")
    if i < 0: return None, False
    j = t.find("{", i)
    if j < 0: return None, True
    d = 0
    for k in range(j, len(t)):
        if t[k] == "{": d += 1
        elif t[k] == "}":
            d -= 1
            if d == 0: return t[j+1:k], False
    return None, True

def ans(t):
    if not t: return None, False
    v, c = boxed(t)
    if v is not None: return v, c
    h = re.findall(r"####\s*(.+)", t)
    if h: return h[-1].strip(), c
    h = re.findall(r"(?:final answer|the answer)\s*(?:is|:)\s*(.+)", t, re.I)
    if h: return h[-1].strip(), c
    return None, c

def norm(a):
    """v2: strips LaTeX thousands separators and commas inside numbers."""
    if a is None: return None
    s = str(a).strip()
    s = s.replace("\\!", "").replace("\\,", "").replace("\\;", "").replace("\\ ", "")
    s = s.replace("\\left", "").replace("\\right", "")
    s = re.sub(r"\\text\{([^}]*)\}", r"\1", s)
    s = re.sub(r"\\mbox\{([^}]*)\}", r"\1", s)
    for p in ("\\dfrac", "\\tfrac"):
        if s.startswith(p): s = "\\frac" + s[len(p):]
    s = s.replace("$", "").replace(" ", "").replace("\n", "")
    s = re.sub(r"(?<=\d),(?=\d\d\d)", "", s)   # 2,880 -> 2880
    s = re.sub(r"\\%$", "", s)
    s = s.rstrip(".").rstrip("$")
    if s.endswith("%"): s = s[:-1]
    if s.startswith("\\(") and s.endswith("\\)"): s = s[2:-2]
    return s.lower()

def agree(a, b):
    na, nb = norm(a), norm(b)
    if na is None or nb is None: return False
    if na == nb: return True
    try: return abs(float(na) - float(nb)) < 1e-6
    except Exception: return False

def ok(text, gold):
    a, _ = ans(text); return agree(a, gold)

NO_ERR = ("no factual error","no error","is correct","solution is correct",
          "no mistakes","looks correct","no issues","correct as written",
          "i agree","agree with the solution","no corrections")
def clean(t):
    x = (t or "").lower(); return any(k in x for k in NO_ERR)



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traces", required=True)
    ap.add_argument("--probed", required=True)
    ap.add_argument("--gens-per-trace", type=int, default=6)
    args = ap.parse_args()

    traces = load_traces(args.traces)
    with open(args.probed, "r", encoding="utf-8") as f:
        probed = json.load(f)

    rate = {}
    for p in probed:
        pid = p.get("pid") or p.get("id")
        if pid is not None and p.get("pass_rate") is not None:
            rate[pid] = float(p["pass_rate"])

    bypid = defaultdict(list)
    for t in traces:
        bypid[t["pid"]].append(t)

    BAR = "=" * 70
    D = "-" * 70
    print(BAR)
    print("COVERAGE GATE: debate vs rejection sampling at MATCHED budget")
    print(BAR)

    rows = []
    for pid, ts in bypid.items():
        if pid not in rate:
            continue
        gold = ts[0]["gold"]
        budget = len(ts) * args.gens_per_trace
        p = rate[pid]
        cov_dbg = any(bool(t.get("final_correct")) for t in ts)
        r1 = []
        for t in ts:
            s = [m for m in t["messages"] if m["role"] == "solver"]
            if s:
                r1.append(ans(s[0]["text"])[0])
        rows.append({
            "pid": pid,
            "p": p,
            "budget": budget,
            "cov_dbg": cov_dbg,
            "cov_r1": any(agree(a, gold) for a in r1),
            "rs": 1.0 - (1.0 - p) ** budget,
        })

    if not rows:
        print("  no pid overlap between traces and probe file")
        return 1

    n = len(rows)
    B = rows[0]["budget"]
    dbg = sum(1 for r in rows if r["cov_dbg"])
    rs = sum(r["rs"] for r in rows)
    print("  problems matched        : %d" % n)
    print("  generations per problem : %d" % B)
    print(D)
    print("  debate coverage         : %d / %d  (%.1f%%)   MEASURED"
          % (dbg, n, 100.0 * dbg / n))
    print("  rejection sampling @%-3d : %.1f / %d  (%.1f%%)   EXPECTED from probe"
          % (B, rs, n, 100.0 * rs / n))
    print("  advantage of debating   : %+.1f problems  (%+.1f pts)"
          % (dbg - rs, 100.0 * (dbg - rs) / n))

    print(D)
    print("  BY DIFFICULTY BAND")
    print("  %-12s %5s %8s %9s %9s"
          % ("band", "n", "mean p", "debate", "RS@%d" % B))
    bands = [(0.0, 0.0), (0.001, 0.10), (0.10, 0.30),
             (0.30, 0.60), (0.60, 0.90), (0.90, 1.0)]
    for lo, hi in bands:
        if lo == 0.0 and hi == 0.0:
            sel = [r for r in rows if r["p"] == 0.0]
            lab = "p == 0"
        else:
            sel = [r for r in rows if lo < r["p"] <= hi]
            lab = "%.2f-%.2f" % (lo, hi)
        if not sel:
            continue
        d = sum(1 for r in sel if r["cov_dbg"]) / len(sel)
        s = sum(r["rs"] for r in sel) / len(sel)
        mp = sum(r["p"] for r in sel) / len(sel)
        print("  %-12s %5d %8.3f %8.1f%% %8.1f%%"
              % (lab, len(sel), mp, 100.0 * d, 100.0 * s))

    only = [r for r in rows if r["cov_dbg"] and r["rs"] < 0.5]
    print(D)
    print("  DEBATE-ONLY WINS: solved by the debate, while rejection sampling")
    print("  at the same budget would find them less than half the time.")
    print("    count : %d / %d  (%.1f%%)" % (len(only), n, 100.0 * len(only) / n))
    for r in sorted(only, key=lambda x: x["p"])[:12]:
        print("      %-14s probe p=%.3f  RS@%d=%.2f"
              % (r["pid"], r["p"], B, r["rs"]))

    print(D)
    if dbg - rs > 0.05 * n:
        print("  VERDICT: the debate reaches problems rejection sampling cannot")
        print("  at matched budget. The data engine is real. Proceed.")
    elif dbg - rs > 0:
        print("  VERDICT: debate is ahead, but by less than 5 points. Weak.")
        print("  Get a harder pool before building on this.")
    else:
        print("  VERDICT: rejection sampling matches or beats the debate at the")
        print("  same cost. The debate is not earning its budget as a data")
        print("  engine on this pool. Change the pool, not the estimator.")
    print(BAR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
