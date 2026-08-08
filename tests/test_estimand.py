"""
Proof that the current estimand cannot produce a non-zero answer.

The central test is `test_redundancy_masking`. It builds a debate in which one
critique is unambiguously decisive -- without it the final answer is wrong,
with it the final answer is right -- and then shows:

    controlled direct effect (what replay.py measures)  ->  delta = 0.0
    total effect            (what estimands.py adds)    ->  delta = 1.0

Same trace, same backend, same k. The difference is entirely the estimand.

This is the experiment that separates "my benchmark is saturated" from "my
measurement is structurally blind". Both are true of your run, but only the
second survives moving to a harder dataset, which is why it has to be fixed
first. This figure belongs in the thesis.

Run:  python -m pytest tests/test_estimand.py -v -s
"""

import asyncio
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.debate.schema import Message, Trace
from src.counterfactual.replay import trace_utilities
from src.counterfactual.estimands import trace_utilities_total, ablation_surrogate
from src.analysis.stats import (
    paired_bootstrap_ci,
    eb_shrink,
    required_k,
    benjamini_hochberg,
)
from src.selection.causal import CausalSelector
from src.selection.random_lenmatched import RandomLenMatchedSelector
from src.selection.last_round_only import LastRoundOnlySelector


GOLD = "42"
WRONG = "99"

SOLVER_1 = f"Step 1: set up the equation. SUBTLE_ERROR: I dropped a sign here. Therefore ANSWER={WRONG}. \\boxed{{{WRONG}}}"
CRITIC_1 = f"CRITIC_DECISIVE: the sign in step 1 is wrong. Redo it; the answer should be {GOLD}."
SOLVER_2 = f"Corrected the sign as instructed. Therefore ANSWER={GOLD}. \\boxed{{{GOLD}}}"
CRITIC_2 = "This looks correct. No further issues."
SOLVER_3 = f"Final restatement of the full solution. Therefore ANSWER={GOLD}. \\boxed{{{GOLD}}}"
VERIFIER = f"\\boxed{{{GOLD}}}"


def make_trace() -> Trace:
    """solver -> critic -> solver -> critic -> solver -> verifier, as your harness emits."""
    msgs = [
        Message("r1.solver", 1, "solver", SOLVER_1, answer=WRONG, parents=[]),
        Message("r1.critic", 1, "critic", CRITIC_1, answer=None, parents=["r1.solver"]),
        Message("r2.solver", 2, "solver", SOLVER_2, answer=GOLD, parents=["r1.solver", "r1.critic"]),
        Message("r2.critic", 2, "critic", CRITIC_2, answer=None, parents=["r2.solver"]),
        Message("r3.solver", 3, "solver", SOLVER_3, answer=GOLD, parents=["r2.solver", "r2.critic"]),
        Message("r4.verifier", 4, "verifier", VERIFIER, answer=GOLD,
                parents=["r1.solver", "r1.critic", "r2.solver", "r2.critic", "r3.solver"]),
    ]
    return Trace(
        pid="mock_0001",
        trace_id="mock_0001:s0",
        question="What is the answer?",
        gold=GOLD,
        messages=msgs,
        final_answer=GOLD,
        final_correct=True,
    )


class RedundantMockBackend:
    """
    A deliberately simple debate world with two realistic properties:

    1. The verifier is EXTRACTIVE: it reports the most recent answer it can
       see. That is what a verifier actually does once the last solver has
       restated a complete solution -- exactly what your r4.verifier did on
       math_0093.

    2. Errors are detectable from a surface cue (SUBTLE_ERROR) that survives
       only in the original wording. Once a solution is rewritten, the same
       flaw is no longer caught. This models the documented fact that critics
       fire on textual signals rather than by re-deriving the solution.

    Together these make r1.critic causally decisive while pinning the direct
    effect at exactly zero.
    """

    def __init__(self):
        self.calls = 0

    async def generate(self, messages, n=1, temperature=0.7, max_tokens=None, cache_nonce=None):
        self.calls += 1
        blob = "\n".join(m["content"] for m in messages)

        # verifier: report the latest visible answer
        if any(m.get("role") == "system" and "verifier" in m["content"] for m in messages):
            found = re.findall(r"ANSWER=(\d+)", blob)
            val = found[-1] if found else "0"
            return [f"\\boxed{{{val}}}"] * n

        # revision (check first: both templates embed the solution)
        if "Here is feedback/critique on that solution:" in blob:
            if "CRITIC_DECISIVE" in blob:
                return [SOLVER_2] * n
            return [f"No changes were suggested, so the solution stands. ANSWER={WRONG}. \\boxed{{{WRONG}}}"] * n

        # critique
        if "Now, critically analyze this solution." in blob:
            if "SUBTLE_ERROR" in blob:
                return [CRITIC_1] * n
            return [CRITIC_2] * n

        # fresh solve
        return [SOLVER_1] * n


# ------------------------------------------------------------------ the point


def test_redundancy_masking():
    """Direct effect calls the decisive critique worthless. Total effect does not."""
    direct = asyncio.run(trace_utilities(make_trace(), RedundantMockBackend(), k=4))
    total = asyncio.run(trace_utilities_total(make_trace(), RedundantMockBackend(), k=4))

    d = {r.mid: r for r in direct}
    t = {r.mid: r for r in total}

    print("\n  mid            direct     total")
    for mid in ["r1.solver", "r1.critic", "r2.solver", "r2.critic", "r3.solver"]:
        dd = d[mid].delta if mid in d else float("nan")
        tt = t[mid].delta if mid in t else float("nan")
        print(f"  {mid:<13} {dd:+.3f}    {tt:+.3f}")

    assert d["r1.critic"].delta == pytest.approx(0.0), "expected the CDE to be blind to r1.critic"
    assert d["r1.critic"].p_factual == pytest.approx(1.0)
    assert d["r1.critic"].p_ablated == pytest.approx(1.0)

    assert t["r1.critic"].delta > 0.9, f"total effect should recover it, got {t['r1.critic'].delta}"
    assert t["r1.critic"].estimand == "total"
    assert t["r1.critic"].n_regenerated > 0


def test_direct_effect_is_zero_for_every_mediated_message():
    """Not a one-off: the CDE is ~0 for every message the last solver supersedes."""
    direct = asyncio.run(trace_utilities(make_trace(), RedundantMockBackend(), k=4))
    mediated = [r for r in direct if r.mid in {"r1.solver", "r1.critic", "r2.solver", "r2.critic"}]
    assert len(mediated) == 4
    assert all(r.delta == pytest.approx(0.0) for r in mediated)
    assert all(r.se == pytest.approx(0.0) for r in mediated), (
        "zero variance as well as zero mean: degeneracy, not sampling noise"
    )


def test_root_ablation_is_flagged_degenerate():
    """Deleting the root collapses the trace. Report that, do not launder it."""
    total = asyncio.run(trace_utilities_total(make_trace(), RedundantMockBackend(), k=2))
    root = next(r for r in total if r.mid == "r1.solver")
    assert root.degenerate is True
    assert "collapses" in root.notes


def test_surrogate_recovers_positive_weight_under_regeneration():
    """Subset ablation + replay gives the decisive critique positive weight."""
    fit = asyncio.run(
        ablation_surrogate(
            make_trace(), RedundantMockBackend(),
            n_ablations=16, k_per_ablation=1, regenerate=True, max_regen_depth=4, seed=0,
        )
    )
    w = dict(zip(fit.mids, fit.weights))
    print(f"\n  surrogate weights: { {k: round(v, 3) for k, v in w.items()} }  r2={fit.r2:.3f}")
    assert w["r1.critic"] > 0.0


# ------------------------------------------------------------------ selectors


def _traces(n=6):
    out = []
    for i in range(n):
        t = make_trace()
        t.pid = f"p{i}"
        t.trace_id = f"p{i}:s0"
        out.append(t)
    return out


def test_causal_selector_does_not_select_everything_on_ties():
    """The old default (min_utility=0.0) kept every message with delta >= 0."""
    traces = _traces()
    utils = {(t.trace_id, m.mid): 0.0 for t in traces for m in t.messages}
    utils[(traces[0].trace_id, "r1.critic")] = 0.6

    sel = CausalSelector(shrink=False, seed=0)
    picked = sel.select(traces, utils, token_budget=100_000)
    flat = [(tid, mid) for tid, mids in picked.items() for mid in mids]

    assert flat == [(traces[0].trace_id, "r1.critic")], f"got {flat}"
    assert sel.stats["n_ties_broken"] > 0


def test_causal_selector_shrinkage_demotes_noisy_estimates():
    """A big-but-noisy delta must not outrank a smaller well-measured one."""
    traces = _traces(2)
    a, b = traces[0].trace_id, traces[1].trace_id
    utils = {(t.trace_id, m.mid): 0.0 for t in traces for m in t.messages}
    ses = {(t.trace_id, m.mid): 0.02 for t in traces for m in t.messages}

    utils[(a, "r1.critic")], ses[(a, "r1.critic")] = 0.50, 0.45   # tiny k, unreliable
    utils[(b, "r1.critic")], ses[(b, "r1.critic")] = 0.30, 0.03   # well measured

    sel = CausalSelector(shrink=True, seed=0)
    picked = sel.select(traces, utils, token_budget=100_000, ses=ses)

    assert b in picked and "r1.critic" in picked[b]
    assert 0.0 <= sel.stats["shrinkage"]["signal_fraction"] <= 1.0


def test_random_lenmatched_actually_matches_length():
    traces = _traces(5)
    reference = {t.trace_id: ["r1.critic", "r3.solver"] for t in traces}
    sel = RandomLenMatchedSelector(seed=1)
    sel.select(traces, {}, token_budget=100_000, reference=reference)

    assert sel.stats["matched"] is True
    assert sel.stats["n_selected"] == sum(len(v) for v in reference.values())
    assert 0.6 <= sel.stats["token_ratio"] <= 1.6, sel.stats


def test_last_round_is_per_trace_and_excludes_verifier():
    traces = _traces(3)
    traces[1].messages = traces[1].messages[:3]  # simulate early termination

    sel = LastRoundOnlySelector()
    picked = sel.select(traces, {}, token_budget=100_000)

    assert sel.stats["n_traces_covered"] == 3, (
        "a globally-computed max round silently drops the short trace"
    )
    assert picked[traces[0].trace_id] == ["r3.solver"]
    assert picked[traces[1].trace_id] == ["r2.solver"]
    for mids in picked.values():
        assert not any("verifier" in m for m in mids)


# ----------------------------------------------------------------- statistics


def test_power_arithmetic_matches_the_budget_reality():
    """Documents why per-message significance at k=8 was never going to work."""
    print(f"\n  k needed for a 25pt effect: {required_k(0.25)}")
    print(f"  k needed for a 10pt effect: {required_k(0.10)}")
    assert required_k(0.25) > 50
    assert required_k(0.10) > 300


def test_paired_bootstrap():
    a = [1.0] * 30 + [0.0] * 20
    b = [1.0] * 24 + [0.0] * 26
    ci = paired_bootstrap_ci(a, b, n_boot=2000, seed=0)
    assert ci.point == pytest.approx(0.12, abs=1e-9)
    assert ci.lo < ci.point < ci.hi


def test_shrinkage_reports_signal_fraction():
    deltas = [0.0] * 40 + [0.4, -0.4]
    ses = [0.25] * 42
    post, diag = eb_shrink(deltas, ses)
    assert 0.0 <= diag["signal_fraction"] <= 1.0
    assert abs(post[-2]) < 0.4, "a lone noisy spike must be pulled toward the mean"


def test_bh_controls_discoveries():
    assert benjamini_hochberg([0.001, 0.02, 0.4, 0.9])[0] is True
    assert benjamini_hochberg([0.3] * 20) == [False] * 20
