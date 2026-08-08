"""
Three estimands for message utility, and why the current one returns zeros.

Background
----------
`replay.py` implements the CONTROLLED DIRECT EFFECT (CDE):

    CDE(m) = P(correct | all messages) - P(correct | all messages except m)

with every other message pinned at its factual text. In an RCR chain

    r1.solver -> r1.critic -> r2.solver -> r2.critic -> r3.solver -> verifier

the causal influence of r1.critic reaches the answer almost entirely THROUGH
r2.solver and r3.solver. Pinning those blocks the mediating path. Worse,
r3.solver restates the complete solution, so the verifier can read the answer
off a descendant no matter what you delete upstream.

Consequence: CDE(m) ~ 0 by construction for every non-terminal message,
regardless of task difficulty. Saturation makes this worse but is NOT the
root cause. A decisive critique on an unsaturated problem still scores 0.
See tests/test_estimand.py for a executable demonstration.

This module adds the two estimands that are not structurally degenerate.

1. total_effect       -- delete m, then REGENERATE every descendant.
                         Measures what the message actually contributed.
                         Cost: k * (1 + n_descendants) generations per message.

2. ablation_surrogate -- delete random SUBSETS, fit a linear model of outcome
                         on presence indicators (ContextCite-style). Fixed cost
                         S regardless of message count, and it recovers
                         interactions that leave-one-out cannot see.

3. student_utility    -- measure the effect on the SMALL MODEL YOU WILL TRAIN,
                         not on the teacher. See student_utility.py.

Known bias, stated on purpose
-----------------------------
Regenerating descendants moves the trace off the factual distribution: the
replacement r2.solver is a fresh draw, not the observed one. That is the
removal-induced distribution shift discussed in the credit-assignment
literature. We do not hide it -- every result carries `degenerate` and
`n_regenerated` so the bias is auditable.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict, field
from typing import Any, Callable

import numpy as np

from ..debate.schema import Trace, Message, descendants, topo_order
from .replay import render_verifier_messages, terminal_mid

try:
    from eval.grade import is_correct
except ImportError:  # allow import when cwd is not the repo root
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from eval.grade import is_correct


@dataclass
class UtilityResult:
    pid: str
    trace_id: str
    mid: str
    role: str
    round: int
    delta: float
    p_factual: float
    p_ablated: float
    k: int
    se: float
    estimand: str
    n_regenerated: int = 0
    degenerate: bool = False
    notes: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------- prompt rebuilding


def _text_of(trace: Trace, mid: str, overrides: dict) -> str:
    if mid in overrides:
        return overrides[mid]
    m = trace.get_message(mid)
    return m.text if m else ""


def _role_of(trace: Trace, mid: str) -> str:
    m = trace.get_message(mid)
    return m.role if m else "?"


def default_node_prompt(
    trace: Trace,
    node: Message,
    overrides: dict,
    dropped: set,
):
    """
    Rebuild the prompt for `node` given regenerated `overrides` and a set of
    `dropped` message ids. Returns None when the node cannot run at all
    (all of its inputs were removed) -- the caller marks that degenerate.

    Mirrors src/debate/prompts.py. If you change the harness prompts, change
    this too, or ablation stops being a valid intervention.
    """
    from ..debate.prompts import (
        get_critique_prompt,
        get_revision_prompt,
        get_solve_prompt,
    )

    live_parents = [p for p in node.parents if p not in dropped]

    if node.role == "verifier":
        return render_verifier_messages(trace, exclude=dropped, overrides=overrides)

    if node.role == "critic":
        solver_parents = [p for p in live_parents if _role_of(trace, p) == "solver"]
        if not solver_parents:
            return None  # nothing left to critique
        target = _text_of(trace, solver_parents[-1], overrides)
        if not target.strip():
            return None
        return [{"role": "user", "content": get_critique_prompt(trace.question, target)}]

    if node.role == "solver":
        if not node.parents:
            return [{"role": "user", "content": get_solve_prompt(trace.question)}]
        solver_parents = [p for p in live_parents if _role_of(trace, p) == "solver"]
        critic_parents = [p for p in live_parents if _role_of(trace, p) == "critic"]
        prev = _text_of(trace, solver_parents[-1], overrides) if solver_parents else ""
        crit = _text_of(trace, critic_parents[-1], overrides) if critic_parents else ""
        if not prev.strip() and not crit.strip():
            return None
        if not crit.strip():
            # Critique deleted: the reviser runs with no feedback. This is the
            # honest "message removed" semantics, and it is exactly where
            # removal-induced distribution shift enters. Flagged by caller.
            crit = "(no feedback was provided)"
        return [{"role": "user", "content": get_revision_prompt(trace.question, prev, crit)}]

    return None


# ------------------------------------------------------------- total effect


async def _p_correct_with_regen(
    trace: Trace,
    dropped: set,
    backend: Any,
    k: int,
    temperature: float,
    max_regen_depth: int | None,
    node_prompt: Callable = default_node_prompt,
    nonce: str = "",
):
    """
    Regenerate every descendant of the dropped set, then score the terminal.

    Returns (p_correct, n_regenerated_nodes, degenerate).

    Runs k independent worlds. Each world regenerates the affected subgraph
    once, so downstream variation is preserved -- that is the whole point of
    the total effect, and the reason it cannot be shared across messages the
    way the factual condition can.
    """
    term = terminal_mid(trace)
    affected: set = set()
    for d in dropped:
        affected |= set(descendants(trace, d))
    affected.discard(term)

    order = topo_order(trace, affected)
    if max_regen_depth is not None:
        order = order[:max_regen_depth]
        # anything beyond the depth cap keeps its factual text

    hits = 0
    degenerate = False
    n_regen = 0

    for w in range(k):
        overrides: dict = {}
        for mid in order:
            node = trace.get_message(mid)
            if node is None:
                continue
            msgs = node_prompt(trace, node, overrides, dropped)
            if msgs is None:
                degenerate = True
                overrides[mid] = ""
                continue
            outs = await backend.generate(
                msgs, n=1, temperature=temperature, cache_nonce=f"{nonce}|w{w}|{mid}"
            )
            overrides[mid] = (outs[0] or "") if outs else ""
            n_regen += 1

        term_node = trace.get_message(term)
        msgs = node_prompt(trace, term_node, overrides, dropped)
        if msgs is None:
            degenerate = True
            continue
        outs = await backend.generate(
            msgs, n=1, temperature=temperature, cache_nonce=f"{nonce}|w{w}|TERM"
        )
        if outs and is_correct(outs[0], trace.gold):
            hits += 1

    return hits / max(k, 1), n_regen, degenerate


async def trace_utilities_total(
    trace: Trace,
    backend: Any,
    k: int = 8,
    temperature: float = 0.7,
    max_regen_depth: int | None = None,
    node_prompt: Callable = default_node_prompt,
):
    """
    Total effect of each non-terminal message: delete it, let the debate re-run
    downstream, and see whether the final answer still lands.

    Cost, RCR chain of 5 non-terminal messages at k worlds:
        factual : k terminal generations (shared across all messages)
        per msg : k * (1 + n_descendants) generations
        total  ~= k * 21 generations per trace, vs k * 6 for the direct effect

    On 30 problems at k=8 that is roughly 5k short generations. Against the
    ~20 minutes you measured for the direct pass, budget 1.5-2 hours. Still an
    overnight free-tier job, and it is the only version of this measurement
    that can return a non-zero answer.
    """
    term = terminal_mid(trace)
    p_fact, _, _ = await _p_correct_with_regen(
        trace, set(), backend, k, temperature, max_regen_depth, node_prompt, nonce="factual"
    )

    out = []
    for m in trace.messages:
        if m.mid == term:
            continue
        p_abl, n_regen, degen = await _p_correct_with_regen(
            trace, {m.mid}, backend, k, temperature, max_regen_depth,
            node_prompt, nonce=f"abl:{m.mid}",
        )
        se = math.sqrt(
            max(p_fact * (1 - p_fact), 0.0) / k + max(p_abl * (1 - p_abl), 0.0) / k
        )
        out.append(
            UtilityResult(
                pid=trace.pid,
                trace_id=trace.trace_id,
                mid=m.mid,
                role=m.role,
                round=m.round,
                delta=p_fact - p_abl,
                p_factual=p_fact,
                p_ablated=p_abl,
                k=k,
                se=se,
                estimand="total",
                n_regenerated=n_regen,
                degenerate=degen,
                notes="root ablation collapses the trace" if degen else "",
            )
        )
    return out


# -------------------------------------------------------- ablation surrogate


@dataclass
class SurrogateFit:
    mids: list
    weights: list
    intercept: float
    r2: float
    n_ablations: int
    holdout_r2: float = 0.0
    masks: list = field(default_factory=list)
    outcomes: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.pop("masks", None)
        d.pop("outcomes", None)
        return d


def _ridge(X: np.ndarray, y: np.ndarray, l2: float):
    n, p = X.shape
    Xb = np.hstack([np.ones((n, 1)), X])
    A = Xb.T @ Xb + l2 * np.eye(p + 1)
    A[0, 0] -= l2  # do not penalise the intercept
    coef = np.linalg.solve(A, Xb.T @ y)
    return coef[1:], float(coef[0])


async def ablation_surrogate(
    trace: Trace,
    backend: Any,
    n_ablations: int = 32,
    k_per_ablation: int = 1,
    keep_prob: float = 0.5,
    temperature: float = 0.7,
    l2: float = 1.0,
    seed: int = 0,
    regenerate: bool = False,
    max_regen_depth: int | None = 2,
) -> SurrogateFit:
    """
    ContextCite-style attribution: ablate random SUBSETS of messages, regress
    correctness on presence indicators, read the coefficients as utilities.

    Why this beats leave-one-out here:
      - fixed cost (n_ablations) instead of O(n_messages)
      - it can see redundancy. Two messages that each restate the answer both
        score ~0 under LOO; the surrogate gives them a shared positive weight
        because the subsets where BOTH are absent do fail.
      - `r2` is a built-in faithfulness check. A low r2 means the linear
        surrogate does not describe this trace and the scores should not be
        used. Report that fraction; it is a real finding, not a failure.

    Set regenerate=True to combine subset ablation with downstream replay
    (recommended). regenerate=False keeps the verifier-only protocol and will
    inherit the redundancy masking described at the top of this file.
    """
    term = terminal_mid(trace)
    mids = [m.mid for m in trace.messages if m.mid != term]
    if not mids:
        return SurrogateFit([], [], 0.0, 0.0, 0)

    rng = np.random.default_rng(seed)
    masks, ys = [], []

    for t in range(n_ablations):
        if t == 0:
            mask = np.ones(len(mids), dtype=int)  # anchor: the factual world
        else:
            mask = (rng.random(len(mids)) < keep_prob).astype(int)
            if mask.sum() == 0:
                mask[rng.integers(0, len(mids))] = 1
        droppedset = {mid for mid, keep in zip(mids, mask) if keep == 0}

        if regenerate:
            p, _, _ = await _p_correct_with_regen(
                trace, droppedset, backend, k_per_ablation, temperature,
                max_regen_depth, default_node_prompt, nonce=f"surr{t}",
            )
        else:
            msgs = render_verifier_messages(trace, exclude=droppedset)
            outs = await backend.generate(
                msgs, n=k_per_ablation, temperature=temperature, cache_nonce=f"surr{t}"
            )
            p = sum(1 for s in outs if is_correct(s, trace.gold)) / max(len(outs), 1)

        masks.append([int(x) for x in mask])
        ys.append(float(p))

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

    return SurrogateFit(
        mids=mids,
        weights=[float(x) for x in w],
        intercept=float(b),
        r2=float(r2),
        n_ablations=n_ablations,
        holdout_r2=float(holdout),
        masks=masks,
        outcomes=[float(v) for v in y],
    )
