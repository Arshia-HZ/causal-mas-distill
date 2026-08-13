#!/usr/bin/env python3
"""
00g_diagnose_signal.py (v3) -- pre-dataset quality gate for debate traces.

WHY v3 (2026-08-14)
-------------------
v2 had two measurement bugs that together manufactured a fake "debate loses
to a single sample" verdict on valid v3 data:

1. CRITIC QUALITY used substring heuristics ("no factual error", "is correct",
   ...) written for v1/v2 prompt style. rcr_v3 critiques are terse, derive
   their own answer, and end in a machine-parseable `VERDICT:` line. They
   almost never contain the old phrases, so v2 counted ~81% of critiques as
   "flags" and produced a nonsense recall/precision table. v3 uses
   parse_verdict() on the VERDICT line.

2. THE GATE compared PER-TRACE debate accuracy against PER-PROBLEM probe
   means. With stratified generation (1 seed for ceiling problems, 3 for the
   rest), per-trace weighting over-represents hard problems 3-to-1 and
   mechanically deflates the debate number. v3 reports both weightings and
   re-weights the single-sample baseline to the same allocation.

No API calls. Runs in seconds.

    python scripts/00g_diagnose_signal.py \
        --traces data/traces_v3.jsonl --probed data/probed_all.json
"""
import argparse, json, math, re, sys
from collections import Counter, defaultdict


def load_rows(path):
    raw = open(path, encoding='utf-8').read().strip()
    if not raw:
        return []
    if raw[0] == '[':
        return json.loads(raw)
    return [json.loads(l) for l in raw.splitlines() if l.strip()]


# ---------------------------------------------------------------- graders
def extract_answer(text):
    if not text:
        return ''
    m = list(re.finditer(r'\\boxed\{', text))
    if m:
        i = m[-1].end(); d = 1; j = i
        while j < len(text) and d > 0:
            if text[j] == '{': d += 1
            elif text[j] == '}': d -= 1
            j += 1
        return text[i:j-1].strip()
    m = re.search(r'####\s*(.+)', text)
    if m:
        return m.group(1).strip()
    m = list(re.finditer(r'answer is[:\s]*(.+)', text, re.I))
    return m[-1].group(1).strip().rstrip('.') if m else ''


def _norm(s):
    if not s:
        return ''
    s = str(s)
    for a in (r'\!', r'\,', r'\;', r'\ ', r'\left', r'\right'):
        s = s.replace(a, '')
    s = s.replace('$', '').replace(' ', '').replace('\n', '')
    for w in (r'\text', r'\mbox', r'\mathrm'):
        s = re.sub(re.escape(w) + r'\{([^{}]*)\}', r'\1', s)
    s = s.replace(r'\dfrac', r'\frac').replace(r'\tfrac', r'\frac')
    s = re.sub(r'(?<=\d),(?=\d\d\d)', '', s)
    return s.rstrip('.$%')


def _agree_fallback(a, b):
    a, b = _norm(a), _norm(b)
    if not a or not b:
        return False
    if a == b:
        return True
    try:
        return abs(float(a) - float(b)) < 1e-6
    except Exception:
        return False


try:  # prefer the repo grader when importable (math_verify-backed)
    sys.path.insert(0, __file__.rsplit('/scripts/', 1)[0])
    from eval.grade import is_correct as _grade_correct  # type: ignore
    def agree(a, b):
        try:
            return bool(_grade_correct(a, b))
        except Exception:
            return _agree_fallback(a, b)
    GRADER = 'eval.grade.is_correct'
except Exception:
    agree = _agree_fallback
    GRADER = 'builtin normalizer (eval.grade not importable)'


def parse_verdict(text):
    """Local copy of src.debate.prompts.parse_verdict (kept import-free)."""
    for line in reversed((text or '').strip().splitlines()):
        s = line.strip().upper()
        if s.startswith('VERDICT:'):
            if 'DISPUTE' in s:
                return 'dispute'
            if 'AGREE' in s:
                return 'agree'
    return 'unparsed'


def band_of(p):
    if p >= 0.999:
        return 'ceiling'
    if p <= 0.001:
        return 'floor'
    return 'headroom'


# ---------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--traces', required=True)
    ap.add_argument('--probed', default=None)
    ap.add_argument('--cap-tokens', type=int, default=4096,
                    help='the max_completion_tokens the traces were generated '
                         'with (NOT the stale 768). Drives the truncation heuristic.')
    ap.add_argument('--chars-per-token', type=float, default=3.1)
    a = ap.parse_args()

    traces = load_rows(a.traces)
    probed = {r['pid']: float(r.get('pass_rate', 0.0))
              for r in load_rows(a.probed)} if a.probed else {}

    print('=' * 66)
    print('SIGNAL DIAGNOSIS v3  (%d traces)' % len(traces))
    print('grader :', GRADER)
    print('cap    : %d tokens (heuristic char cap = %d)'
          % (a.cap_tokens, int(a.cap_tokens * a.chars_per_token)))
    print('=' * 66)

    # ---- completeness / truncation --------------------------------------
    cap_chars = a.cap_tokens * a.chars_per_token
    sol_noans = ver_noans = near_cap = n_msgs = 0
    short_traces = []
    for t in traces:
        msgs = t.get('messages', [])
        n_msgs += len(msgs)
        if len(msgs) < 6:
            short_traces.append(t.get('trace_id', t.get('pid', '?')))
        for m in msgs:
            txt = m.get('text', '') or ''
            if len(txt) > 0.95 * cap_chars:
                near_cap += 1
            if m.get('role') == 'solver' and not extract_answer(txt):
                sol_noans += 1
            if m.get('role') == 'verifier' and not extract_answer(txt):
                ver_noans += 1
    n_sol = sum(1 for t in traces for m in t.get('messages', [])
                if m.get('role') == 'solver')
    n_ver = sum(1 for t in traces for m in t.get('messages', [])
                if m.get('role') == 'verifier')
    print('COMPLETENESS')
    print('  solver msgs with no answer  : %d/%d (%.1f%%)   [gate: <= 5%%]'
          % (sol_noans, n_sol, 100 * sol_noans / max(1, n_sol)))
    print('  verifier msgs with no answer: %d/%d (%.1f%%)   [gate: <= 1%%]'
          % (ver_noans, n_ver, 100 * ver_noans / max(1, n_ver)))
    print('  messages near %d-token cap   : %d/%d (%.1f%%)'
          % (a.cap_tokens, near_cap, n_msgs, 100 * near_cap / max(1, n_msgs)))
    print('  short traces (<6 msgs)      : %d   %s'
          % (len(short_traces), '(resilient-retry degradation; builder drops them)'
             if short_traces else ''))
    print('-' * 66)

    # ---- seed independence -----------------------------------------------
    by_pid = defaultdict(list)
    for t in traces:
        by_pid[t['pid']].append(t)
    multi = {p: ts for p, ts in by_pid.items() if len(ts) > 1}
    ident = 0
    for p, ts in multi.items():
        r1 = [(m.get('text', '') or '').strip()
              for t in ts
              for m in t.get('messages', [])
              if m.get('role') == 'solver' and m.get('round') == 1]
        if len(r1) > 1 and len(set(r1)) < len(r1):
            ident += 1
    print('SEED INDEPENDENCE over %d multi-seed problems' % len(multi))
    print('  problems with identical round-1 seeds: %d   [gate: 0]' % ident)
    print('-' * 66)

    # ---- structure (absorbs everything useful 00f_check_traces.py did) ----
    msgs_hist = Counter()
    roles = Counter()
    for t in traces:
        msgs_hist[len(t.get('messages', []))] += 1
        for m in t.get('messages', []):
            roles[m.get('role', '?')] += 1
    seeds_hist = Counter(len(ts) for ts in by_pid.values())
    spread = Counter()
    for p, ts in by_pid.items():
        accs = []
        for t in ts:
            af = t.get('final_answer') or ''
            accs.append(bool(af) and agree(af, t.get('gold', '')))
        spread[round(sum(accs) / len(accs), 3)] += 1
    print('STRUCTURE')
    print('  messages per trace :', dict(sorted(msgs_hist.items())))
    print('  roles              :', dict(roles))
    print('  seeds per problem  :', dict(sorted(seeds_hist.items())))
    print('  per-pid acc spread :', dict(sorted(spread.items())))
    print('-' * 66)

    # ---- does the debate move the answer? (one grader both sides) --------
    moved = rescued = broken = scored = 0
    r1_ok = fin_ok = 0
    for t in traces:
        msgs = t.get('messages', [])
        gold = t.get('gold', '')
        r1 = next((m for m in msgs if m.get('role') == 'solver'
                   and m.get('round') == 1), None)
        if not r1:
            continue
        a1 = extract_answer(r1.get('text', ''))
        af = t.get('final_answer') or ''
        if not a1 or not af:
            continue
        scored += 1
        c1 = agree(a1, gold)
        cf = agree(af, gold)
        r1_ok += c1
        fin_ok += cf
        if _norm(a1) != _norm(af):
            moved += 1
            if cf and not c1:
                rescued += 1
            elif c1 and not cf:
                broken += 1
    print('DOES THE DEBATE MOVE THE ANSWER?  (re-graded, one grader)')
    print('  traces scored     :', scored)
    print('  moved             : %d (%.1f%%)' % (moved, 100 * moved / max(1, scored)))
    print('  rescued / broken  : %d / %d   (net %+d)   [gate: rescue >= 2x broken]'
          % (rescued, broken, rescued - broken))
    print('  r1 acc -> final   : %.3f -> %.3f  (%+.3f)'
          % (r1_ok / max(1, scored), fin_ok / max(1, scored),
             (fin_ok - r1_ok) / max(1, scored)))
    file_fc = sum(1 for t in traces if t.get('final_correct'))
    print('  file-recorded final_correct: %.3f  (informational)'
          % (file_fc / max(1, len(traces))))
    print('-' * 66)

    # ---- critic quality via VERDICT lines --------------------------------
    crit = Counter()
    for t in traces:
        msgs = t.get('messages', [])
        gold = t.get('gold', '')
        for i, m in enumerate(msgs):
            if m.get('role') != 'critic':
                continue
            prev = next((mm for mm in reversed(msgs[:i])
                         if mm.get('role') == 'solver'), None)
            if not prev:
                continue
            wrong = not agree(extract_answer(prev.get('text', '')), gold)
            v = parse_verdict(m.get('text', ''))
            crit['n'] += 1
            crit[v] += 1
            if v in ('dispute', 'agree'):
                key = ('wrong' if wrong else 'right') + '_' + v
                crit[key] += 1
    n_c = max(1, crit['n'])
    recall = crit['wrong_dispute'] / max(1, crit['wrong_dispute'] + crit['wrong_agree'])
    precision = crit['wrong_dispute'] / max(1, crit['wrong_dispute'] + crit['right_dispute'])
    print('CRITIC QUALITY  (VERDICT-line based)')
    print('  critiques scored  :', crit['n'])
    print('  dispute / agree / unparsed : %d / %d / %d  (unparsed %.1f%%  [warn if >10%%])'
          % (crit['dispute'], crit['agree'], crit['unparsed'],
             100 * crit['unparsed'] / n_c))
    print('  solver WRONG, disputed (recall)    : %d/%d = %.3f   [gate: >= 0.30]'
          % (crit['wrong_dispute'],
             crit['wrong_dispute'] + crit['wrong_agree'], recall))
    print('  precision when disputing           : %d/%d = %.3f'
          % (crit['wrong_dispute'],
             crit['wrong_dispute'] + crit['right_dispute'], precision))
    print('-' * 66)

    # ---- the honest debate-vs-single-sample comparison -------------------
    print('DEBATE vs SINGLE SAMPLE  (like-for-like weightings)')
    if probed:
        bands = defaultdict(lambda: dict(pids=0, traces=0, p_sum=0.0,
                                         deb_pid=[], deb_tr=[]))
        for p, ts in by_pid.items():
            if p not in probed:
                continue
            b = band_of(probed[p])
            d = bands[b]
            d['pids'] += 1
            d['p_sum'] += probed[p]
            d['traces'] += len(ts)
            accs = []
            for t in ts:
                af = t.get('final_answer') or ''
                c = bool(af) and agree(af, t.get('gold', ''))
                d['deb_tr'].append(c)
                accs.append(c)
            d['deb_pid'].append(sum(accs) / len(accs))

        hdr = '%-9s %5s %6s | %7s %8s | %8s %8s' % (
            'band', 'pids', 'traces', 'RS/pid', 'deb/pid', 'RS/trc*', 'deb/trc')
        print(' ', hdr)
        print(' ', '-' * (len(hdr) + 1))
        tot = dict(pids=0, traces=0, p_sum=0.0, dp=[], dt=[], wrs=0.0)
        for b in ('ceiling', 'headroom', 'floor'):
            d = bands.get(b)
            if not d or not d['pids']:
                continue
            rs_pid = d['p_sum'] / d['pids']
            deb_pid = sum(d['deb_pid']) / len(d['deb_pid'])
            rs_tr = sum(probed[p] * len(by_pid[p]) for p in by_pid
                        if p in probed and band_of(probed[p]) == b) / d['traces']
            deb_tr = sum(d['deb_tr']) / len(d['deb_tr'])
            print('  %-9s %5d %6d | %7.3f %8.3f | %8.3f %8.3f'
                  % (b, d['pids'], d['traces'], rs_pid, deb_pid, rs_tr, deb_tr))
            tot['pids'] += d['pids']; tot['traces'] += d['traces']
            tot['p_sum'] += d['p_sum']; tot['dp'] += d['deb_pid']
            tot['dt'] += d['deb_tr']
            tot['wrs'] += sum(probed[p] * len(by_pid[p]) for p in by_pid
                              if p in probed and band_of(probed[p]) == b)
        print(' ', '-' * (len(hdr) + 1))
        print('  %-9s %5d %6d | %7.3f %8.3f | %8.3f %8.3f' % (
            'ALL', tot['pids'], tot['traces'],
            tot['p_sum'] / max(1, tot['pids']),
            sum(tot['dp']) / max(1, len(tot['dp'])),
            tot['wrs'] / max(1, tot['traces']),
            sum(tot['dt']) / max(1, len(tot['dt']))))
        print('  * RS/trc re-weights single-sample accuracy to the SAME 1-vs-3')
        print('    seed allocation as the trace file. v2 skipped this and')
        print('    manufactured a fake "debate loses" verdict.')
        cov = sum(1 for p, ts in by_pid.items()
                  if any((t.get('final_answer') or '') and
                         agree(t['final_answer'], t.get('gold', '')) for t in ts))
        print('  coverage (pids with >=1 correct trace): %d/%d   [gate: >= 600]'
              % (cov, len(by_pid)))
    else:
        print('  (no --probed file; skipping baseline comparison)')
        cov = 0
    print('=' * 66)

    # ---- gate ------------------------------------------------------------
    checks = [
        ('coverage >= 600 pids', cov >= 600 if probed else None),
        ('solver answerless <= 5%', sol_noans / max(1, n_sol) <= 0.05),
        ('verifier answerless <= 1%', ver_noans / max(1, n_ver) <= 0.01),
        ('verdicts parsed >= 90%', crit['unparsed'] / n_c <= 0.10),
        ('seeds independent', ident == 0),
        ('rescue >= 2x broken', rescued >= 2 * max(1, broken)),
        ('critic recall >= 0.30', recall >= 0.30),
    ]
    print('GATE')
    ok = True
    for name, res in checks:
        if res is None:
            print('  [skip]', name); continue
        print('  [%s] %s' % ('PASS' if res else 'FAIL', name))
        ok = ok and res
    print()
    print('VERDICT:', 'GO -- build Arm A, then A/B/C, then dry-run, then train.'
          if ok else
          'NO-GO -- fix the failing check BEFORE spending GPU hours.')


if __name__ == '__main__':
    main()
