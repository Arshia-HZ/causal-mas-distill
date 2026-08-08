"""
Student-Conditioned Message Utility (SCMU).

The reframe
-----------
Your current estimand asks: "did this message change the TEACHER's answer?"
For a distillation-data selector that is the wrong question, and your own
numbers prove it: 676 of 785 MATH problems sit at teacher pass-rate 1.0, so
the teacher-side outcome variable is constant and no selector can be built
on it.

The question you actually need answered is:

    "does this message change what the STUDENT can do?"

    U_student(m) = P_student(correct | context with m)
                 - P_student(correct | context without m)

Why this is the right move, not just a cheaper one:

1. Non-degenerate by construction. A 1.5B student is nowhere near ceiling on
   the problems where a frontier teacher is. The saturation that killed the
   teacher-side measurement is exactly the regime where student-side utility
   is largest.

2. Aligned with the downstream objective. You select data to change the
   student. In-context marginal contribution is a zeroth-order proxy for the
   in-weights training effect, so the estimand and the goal finally match.

3. Free. The student runs locally on a T4. No API cap, no per-token cost, so
   k=32 or k=64 becomes affordable and the SE drops from 0.25 to 0.09-0.13.
   This is the difference between an unmeasurable and a measurable effect.

4. Novel. Agent-level LOO attribution, per-decision credit assignment, and
   causal CoT-step pruning all score messages against the PRODUCER. Scoring
   them against the RECEIVER-TO-BE-TRAINED is the dyadic framing, and it is
   not occupied.

The validation you must run (do not skip)
-----------------------------------------
SCMU is a proxy. Prove it tracks the thing it proxies for:
`bridge_correlation()` compares in-context utility rank against actual
retraining effect on a small subset. Report Spearman rho. Without that number
a reviewer will call this a heuristic, and they will be right.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Sequence

import numpy as np

from ..debate.schema import Trace
from .estimands import UtilityResult, _ridge

try:
    from eval.grade import is_correct
except ImportError:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from eval.grade import is_correct


HINT_HEADER = (
    "You are solving a problem. Below are notes from a discussion between "
    "other solvers. Some notes may be irrelevant or wrong. Use whatever helps."
)
HINT_INSTRUCTION = (
    "Now solve the problem yourself. Show brief reasoning and put the final "
    "answer in \\boxed{}."
)


def render_student_prompt(
    trace: Trace,
    include_mids: Sequence[str],
    char_cap_per_message: int = 3000,
) -> list[dict]:
    """
    Build the student's prompt from a subset of debate messages.

    Deliberately excludes the verifier: it states the final answer outright,
    so including it makes every other message look useless. Keeping it in is
    the single easiest way to accidentally re-derive "only the last message
    matters", which is a leakage artefact, not a finding.
    """
    keep = set(include_mids)
    parts: list[dict] = [
        {"role": "system", "content": HINT_HEADER},
        {"role": "user", "content": f"Problem:\n{trace.question}"},
    ]
    for m in trace.messages:
        if m.role == "verifier":
            continue
        if m.mid not in keep:
            continue
        body = m.text if len(m.text) <= char_cap_per_message else m.text[:char_cap_per_message] + "\n[...]"
        parts.append({"role": "user", "content": f"[note {m.mid}]\n{body}"})
    parts.append({"role": "user", "content": HINT_INSTRUCTION})
    return parts


async def _student_p_correct(
    trace: Trace,
    include_mids: Sequence[str],
    student: Any,
    k: int,
    temperature: float,
    nonce: str,
) -> float:
    msgs = render_student_prompt(trace, include_mids)
    outs = await student.generate(msgs, n=k, temperature=temperature, cache_nonce=nonce)
    if not outs:
        return 0.0
    return sum(1 for s in outs if is_correct(s, trace.gold)) / len(outs)


async def student_message_utilities(
    trace: Trace,
    student: Any,
    k: int = 32,
    temperature: float = 0.8,
    include_baseline: bool = True,
) -> list[UtilityResult]:
    """
    Leave-one-out student-conditioned utility for every non-verifier message.

    Cost: (1 + n_messages) * k local generations. On a T4 with a 1.5B student
    and short answers this is minutes, not hours, and k=32 is affordable.

    `include_baseline` also measures the no-notes condition, which gives you
    the trace-level headroom: if the student already solves the problem with
    zero notes, no message in that trace can have positive utility and the
    trace should be dropped from the selection pool. Report how many traces
    that removes -- it is the student-side analogue of your 676/785 ceiling
    statistic, and it belongs in the paper.
    """
    mids = [m.mid for m in trace.messages if m.role != "verifier"]
    if not mids:
        return []

    p_full = await _student_p_correct(trace, mids, student, k, temperature, "stu:full")

    p_none = None
    if include_baseline:
        p_none = await _student_p_correct(trace, [], student, k, temperature, "stu:none")

    out: list[UtilityResult] = []
    for m in trace.messages:
        if m.role == "verifier":
            continue
        subset = [x for x in mids if x != m.mid]
        p_abl = await _student_p_correct(trace, subset, student, k, temperature, f"stu:-{m.mid}")
        se = math.sqrt(
            max(p_full * (1 - p_full), 0.0) / k + max(p_abl * (1 - p_abl), 0.0) / k
        )
        note = ""
        if p_none is not None:
            note = f"p_no_notes={p_none:.3f}"
            if p_none >= 0.95:
                note += " | STUDENT_CEILING: drop this trace"
        out.append(
            UtilityResult(
                pid=trace.pid,
                trace_id=trace.trace_id,
                mid=m.mid,
                role=m.role,
                round=m.round,
                delta=p_full - p_abl,
                p_factual=p_full,
                p_ablated=p_abl,
                k=k,
                se=se,
                estimand="student_loo",
                notes=note,
            )
        )
    return out


async def student_surrogate_utilities(
    trace: Trace,
    student: Any,
    n_ablations: int = 32,
    k_per_ablation: int = 4,
    keep_prob: float = 0.5,
    temperature: float = 0.8,
    l2: float = 1.0,
    seed: int = 0,
) -> tuple[list[UtilityResult], dict]:
    """
    Subset-ablation surrogate, student side. Preferred over LOO once traces
    have more than ~6 messages: fixed cost, and it survives redundancy.

    Returns (utilities, fit_diagnostics). Check `holdout_r2` before trusting
    the weights; traces where the linear surrogate does not fit should be
    reported as a coverage limitation rather than silently scored.
    """
    mids = [m.mid for m in trace.messages if m.role != "verifier"]
    if not mids:
        return [], {"r2": 0.0, "holdout_r2": 0.0, "n_ablations": 0}

    rng = np.random.default_rng(seed)
    masks, ys = [], []
    for t in range(n_ablations):
        if t == 0:
            mask = np.ones(len(mids), dtype=int)
        elif t == 1:
            mask = np.zeros(len(mids), dtype=int)  # anchor the no-notes end
        else:
            mask = (rng.random(len(mids)) < keep_prob).astype(int)
        subset = [mid for mid, keep in zip(mids, mask) if keep == 1]
        p = await _student_p_correct(
            trace, subset, student, k_per_ablation, temperature, f"stusurr:{t}"
        )
        masks.append([int(x) for x in mask])
        ys.append(p)

    X = np.asarray(masks, dtype=float)
    y = np.asarray(ys, dtype=float)
    w, b = _ridge(X, y, l2)

    pred = X @ w + b
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 0.0

    holdout = 0.0
    if len(y) >= 8:
        half = len(y) // 2
        w1, b1 = _ridge(X[:half], y[:half], l2)
        p2 = X[half:] @ w1 + b1
        sr = float(((y[half:] - p2) ** 2).sum())
        st = float(((y[half:] - y[half:].mean()) ** 2).sum())
        holdout = 1.0 - sr / st if st > 1e-12 else 0.0

    # SE of a ridge coefficient under homoscedastic residual noise
    n, p_dim = X.shape
    resid_var = ss_res / max(n - p_dim - 1, 1)
    try:
        cov = np.linalg.inv(X.T @ X + l2 * np.eye(p_dim)) * resid_var
        coef_se = np.sqrt(np.clip(np.diag(cov), 0, None))
    except np.linalg.LinAlgError:
        coef_se = np.full(p_dim, float("nan"))

    index = {m.mid: m for m in trace.messages}
    out = []
    for i, mid in enumerate(mids):
        m = index[mid]
        out.append(
            UtilityResult(
                pid=trace.pid,
                trace_id=trace.trace_id,
                mid=mid,
                role=m.role,
                round=m.round,
                delta=float(w[i]),
                p_factual=float(y[0]),
                p_ablated=float(y[1]) if len(y) > 1 else 0.0,
                k=k_per_ablation * n_ablations,
                se=float(coef_se[i]),
                estimand="student_surrogate",
                notes=f"holdout_r2={holdout:.3f}",
            )
        )
    return out, {"r2": r2, "holdout_r2": holdout, "n_ablations": n_ablations, "intercept": float(b)}


def bridge_correlation(
    incontext_utility: dict[tuple[str, str], float],
    retrain_effect: dict[tuple[str, str], float],
) -> dict:
    """
    Spearman correlation between in-context utility (cheap) and measured
    retraining effect (expensive) over a shared subset of messages.

    This single number decides whether SCMU is a method or a heuristic.
    Target it on ~150-200 messages; that is affordable and enough for a
    meaningful rho.
    """
    keys = sorted(set(incontext_utility) & set(retrain_effect))
    if len(keys) < 3:
        return {"n": len(keys), "spearman": float("nan"), "note": "need >=3 shared messages"}

    a = np.array([incontext_utility[k] for k in keys], dtype=float)
    b = np.array([retrain_effect[k] for k in keys], dtype=float)

    def _rank(v: np.ndarray) -> np.ndarray:
        order = v.argsort()
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(len(v), dtype=float)
        # average ties
        _, inv, counts = np.unique(v, return_inverse=True, return_counts=True)
        sums = np.zeros(len(counts))
        np.add.at(sums, inv, r)
        return (sums / counts)[inv]

    ra, rb = _rank(a), _rank(b)
    ra -= ra.mean()
    rb -= rb.mean()
    denom = math.sqrt(float((ra**2).sum()) * float((rb**2).sum()))
    rho = float((ra * rb).sum() / denom) if denom > 0 else float("nan")
    return {"n": len(keys), "spearman": rho}
