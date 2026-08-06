"""
Pre-spend gate. Every one of these must pass before a single paid API call.

Run:  python -m pytest tests/ -v
Cost: $0 (MockBackend only).
"""

import asyncio
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backends.mock import MockBackend
from src.debate.schema import Message, Trace
from src.counterfactual import replay as R
from src.render import template as T
from eval.grade import extract_answer, extract_boxed, is_correct

MARK = "__CAUSAL__"


def make_trace(leaky: bool = False) -> Trace:
    """3-round solver/critic/verifier chain. Verifier sees the whole transcript."""
    msgs = [
        Message("r1.solver", 1, "solver", "Initial attempt: I think it is 17.", "17", []),
        Message(
            "r1.critic",
            1,
            "critic",
            f"{MARK} You dropped a factor of two. Recompute the sum.",
            None,
            ["r1.solver"],
        ),
        Message(
            "r2.solver",
            2,
            "solver",
            "Filler restatement with no new information.",
            "42",
            ["r1.solver", "r1.critic"],
        ),
    ]
    parents = (
        ["r1.solver", "r1.critic", "r2.solver"] if not leaky else ["r1.solver"]
    )
    msgs.append(Message("r3.verifier", 3, "verifier", "Final: 42.", "42", parents))
    return Trace(
        pid="p1",
        trace_id="p1:s0",
        question="What is the answer?",
        gold="42",
        messages=msgs,
        final_answer="42",
        final_correct=True,
    )


# ---------------------------------------------------------------- grading


def test_boxed_extraction_with_nested_braces():
    assert extract_boxed(r"so \boxed{\frac{1}{2}} done") == r"\frac{1}{2}"
    assert extract_boxed(r"\boxed{1} then \boxed{2}") == "2"  # last wins
    assert extract_boxed("no box here") is None


def test_grader_handles_real_formats():
    assert is_correct(r"Therefore \boxed{42}.", "42")
    assert is_correct("#### 42", "42")
    assert is_correct("The final answer is 42", "42")
    assert not is_correct(r"\boxed{17}", "42")


def test_grader_never_raises():
    for bad in [None, "", "\\boxed{", "{[(", "\x00"]:
        assert is_correct(bad, "42") in (True, False)


def test_grader_does_not_false_positive_on_midtext_answer():
    txt = "The answer is not obvious. Let me compute.\nFinal answer: 42"
    assert is_correct(txt, "42")


# ------------------------------------------------------ structural guards


def test_parents_invariant_holds_on_wellformed_trace():
    R.assert_parents_invariant(make_trace())


def test_parents_invariant_detects_dangling_parent():
    t = make_trace()
    t.get_message("r2.solver").parents = ["r1.solver", "r9.ghost"]
    with pytest.raises(AssertionError, match="does not exist"):
        R.assert_parents_invariant(t)


def test_terminal_must_see_everything():
    R.assert_terminal_sees_all(make_trace())
    with pytest.raises(AssertionError, match="does not see"):
        R.assert_terminal_sees_all(make_trace(leaky=True))


def test_ablation_actually_removes_the_message():
    t = make_trace()
    full = "\n".join(
        m["content"] for m in R.render_verifier_messages(t, exclude=set())
    )
    abl = "\n".join(
        m["content"] for m in R.render_verifier_messages(t, exclude={"r1.critic"})
    )
    assert MARK in full
    assert MARK not in abl


# --------------------------------------------------------- sample collapse


def test_no_sample_collapse():
    """k samples must not be k copies of one draw."""
    b = MockBackend(p_with=0.5, p_without=0.5)
    out = asyncio.run(b.generate([{"role": "user", "content": "x"}], n=16))
    assert len(out) == 16
    assert len(set(out)) > 1, "cache collapse: all samples identical"
    answers = {extract_answer(s) for s in out}
    assert len(answers) == 2, f"zero variance in outcomes: {answers}"


def test_one_call_per_condition():
    """k=16 must cost ONE request, not 16 -- and not 1 request reused 16x."""
    b = MockBackend()
    asyncio.run(R.trace_utilities(make_trace(), b, k=16))
    assert b.calls == 4, f"expected 1 factual + 3 ablated calls, got {b.calls}"
    assert b.generations == 64


# ------------------------------------------------------- estimator recovery


def test_estimator_recovers_known_causal_structure():
    """
    Ground truth: only r1.critic carries the marker, worth p_with - p_without.
    The estimator must find it, with the right SIGN.
    """
    b = MockBackend(p_with=0.9, p_without=0.2, seed=7)
    us = {u.mid: u for u in asyncio.run(R.trace_utilities(make_trace(), b, k=64))}

    assert us["r1.critic"].delta > 0.4, "failed to detect the causal message"
    assert abs(us["r2.solver"].delta) < 0.25, "false positive on filler message"
    assert abs(us["r1.solver"].delta) < 0.25, "false positive on filler message"

    top = max(us.values(), key=lambda u: u.delta)
    assert top.mid == "r1.critic", "ranking is inverted or wrong"


def test_delta_sign_convention():
    """delta > 0 must mean 'removing it HURT', i.e. the message helped."""
    b = MockBackend(p_with=0.9, p_without=0.1, seed=3)
    us = {u.mid: u for u in asyncio.run(R.trace_utilities(make_trace(), b, k=64))}
    u = us["r1.critic"]
    assert u.p_factual > u.p_ablated
    assert u.delta == pytest.approx(u.p_factual - u.p_ablated)


def test_noise_floor_is_not_identically_zero():
    """tau == 0.0 exactly is the signature of a collapsing cache."""
    b = MockBackend(p_with=0.5, p_without=0.5, seed=11)
    diffs = asyncio.run(R.noise_floor(make_trace(), b, k=16, repeats=4))
    assert len(diffs) == 6
    assert any(d != 0.0 for d in diffs), "placebo diffs all zero -> cache collapse"


# ------------------------------------------------------------- rendering


def test_template_survives_latex_braces():
    """str.format on LaTeX is the single most common crash in this pipeline."""
    t = make_trace()
    t.get_message("r1.critic").text = r"Use \frac{1}{2} and \{x : x>0\} here {oops}"
    out = T.render(t, ["r1.critic"])
    assert r"\frac{1}{2}" in out
    assert "{oops}" in out


def test_target_is_teacher_answer_not_gold():
    t = make_trace()
    t.final_answer = "teacher-said-this"
    out = T.render(t, ["r1.critic"])
    assert out.rstrip().endswith("teacher-said-this")
    assert not out.rstrip().endswith("42")


def test_selection_is_per_trace_and_arms_are_matched():
    """Every arm must yield the same number of examples."""
    traces = [make_trace(), make_trace()]
    traces[1].pid = "p2"
    causal = {"p1:s0": ["r1.critic"], "p2:s0": ["r1.critic"]}
    random_arm = {"p1:s0": ["r2.solver"], "p2:s0": ["r1.solver"]}
    a = T.render_for_sft(traces, causal)
    b = T.render_for_sft(traces, random_arm)
    assert len(a) == len(b) == 2
    assert a[0]["text"] != b[0]["text"]
    assert all(r["prompt"].endswith("\nAnswer:") for r in a)


# ------------------------------------------------------------- integration


def test_harness_generates_valid_trace():
    from src.debate.harness import DebateHarness
    b = MockBackend(gold="42")
    harness = DebateHarness(b, max_rounds=2)
    traces = asyncio.run(harness.run("test1", "What is the answer?", "42", n_solutions=1))
    assert len(traces) == 1
    t = traces[0]
    
    # Must have 4 messages: r1.solver, r1.critic, r2.solver, verifier (since max_rounds=2)
    assert len(t.messages) == 4
    
    assert t.messages[0].mid == "r1.solver"
    assert t.messages[1].mid == "r1.critic"
    assert t.messages[2].mid == "r2.solver"
    assert t.messages[3].mid == "r3.verifier"
    
    assert t.messages[1].parents == ["r1.solver"]
    assert t.messages[2].parents == ["r1.solver", "r1.critic"]
    assert t.messages[3].parents == ["r1.solver", "r1.critic", "r2.solver"]
    
    R.assert_terminal_sees_all(t)
    R.assert_parents_invariant(t)
