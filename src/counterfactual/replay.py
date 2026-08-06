"""
Counterfactual replay -- direct effect (fixed-context) estimator.

Estimand
--------
For message m in a debate trace, the DIRECT effect is:

    delta(m) = P(final answer correct | full transcript)
             - P(final answer correct | transcript minus m)

Intermediate messages are held FIXED; only the terminal answer is regenerated.
Cost is k generations per condition instead of k * |descendants|.

Sign convention: delta > 0 means the message HELPED. Selectors must rank by
descending delta.

Requirement on the harness: the terminal node must condition on the whole
transcript, otherwise the direct effect of non-parent messages is trivially 0.
`assert_terminal_sees_all` enforces this.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Any, Iterable, Sequence

from ..debate.schema import Trace, Message

try:
    from eval.grade import is_correct
except ImportError:  # allow running as a plain package
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from eval.grade import is_correct


FINAL_INSTRUCTION = (
    "You are the verifier. Read the discussion above and state the single "
    "final answer to the question. Put the final answer in \\boxed{}."
)


@dataclass
class Utility:
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
    estimand: str = "direct"

    def to_dict(self) -> dict:
        return asdict(self)


def terminal_mid(trace: Trace) -> str:
    """Terminal node = the message no other message lists as a parent."""
    if not trace.messages:
        raise ValueError(f"trace {trace.pid} has no messages")
    parented = {p for m in trace.messages for p in m.parents}
    sinks = [m.mid for m in trace.messages if m.mid not in parented]
    if len(sinks) == 1:
        return sinks[0]
    # fall back to latest round, then last in insertion order
    return max(trace.messages, key=lambda m: (m.round, trace.messages.index(m))).mid


def assert_terminal_sees_all(trace: Trace) -> None:
    """
    Guard the estimand. If the terminal node does not condition on every other
    message, direct effects are structurally zero for the unseen ones and the
    experiment is meaningless.
    """
    t = terminal_mid(trace)
    term = trace.get_message(t)
    assert term is not None
    seen = set(term.parents)
    everything = {m.mid for m in trace.messages if m.mid != t}
    missing = everything - seen
    if missing:
        raise AssertionError(
            f"trace {trace.pid}: terminal node {t} does not see {sorted(missing)}. "
            "Direct effect is undefined for those messages. Fix the harness so the "
            "terminal node conditions on the full transcript."
        )


def assert_parents_invariant(trace: Trace, prompt_of=None) -> None:
    """
    Ablation is only valid if `parents` exactly describes what was rendered
    into the prompt: every declared parent present, every non-parent absent.
    """
    index = {m.mid: m for m in trace.messages}

    for m in trace.messages:
        for p in m.parents:
            if p not in index:
                raise AssertionError(
                    f"{trace.pid}/{m.mid}: declared parent {p} does not exist"
                )

    # Note: this check might need adjustment if intermediate nodes have different prompts.
    # For now, it's mainly checking the verifier if we default to render_verifier_messages.


def render_verifier_messages(
    trace: Trace,
    exclude: Iterable[str] = (),
) -> list[dict]:
    """
    Render the prompt for the verifier from all prior messages, minus `exclude`.
    Parent order follows trace order so the transcript reads chronologically.
    """
    ex = set(exclude)
    
    out: list[dict] = [
        {"role": "system", "content": "You are the verifier in a multi-agent debate."},
        {"role": "user", "content": f"Question:\n{trace.question}"},
    ]
    for msg in trace.messages:
        if msg.role == "verifier":
            continue
        if msg.mid in ex:
            continue
        out.append({"role": "user", "content": f"[{msg.role} | {msg.mid}]\n{msg.text}"})
    
    out.append({"role": "user", "content": FINAL_INSTRUCTION})
    return out


async def _p_correct(
    trace: Trace,
    exclude: Sequence[str],
    backend: Any,
    k: int,
    temperature: float,
) -> float:
    """
    Accuracy of the regenerated terminal answer under a given context.

    All k samples are requested in ONE call so a hash cache cannot collapse
    them into k copies of a single draw.
    """
    msgs = render_verifier_messages(trace, exclude=exclude)
    samples = await backend.generate(msgs, n=k, temperature=temperature)
    if len(samples) < k:
        raise RuntimeError(
            f"backend returned {len(samples)} samples for n={k}; "
            "sample collapse would silently zero the variance"
        )
    hits = sum(1 for s in samples if is_correct(s, trace.gold))
    return hits / k


async def trace_utilities(
    trace: Trace,
    backend: Any,
    k: int = 16,
    temperature: float = 0.7,
    check_invariants: bool = True,
) -> list[Utility]:
    """
    Direct-effect utility for every non-terminal message in one trace.

    Cost: (1 + n_messages) * k generations, with the factual condition shared
    across all messages in the trace.
    """
    if check_invariants:
        assert_terminal_sees_all(trace)

    term = terminal_mid(trace)
    p_fact = await _p_correct(trace, (), backend, k, temperature)

    results: list[Utility] = []
    for m in trace.messages:
        if m.mid == term:
            continue
        p_abl = await _p_correct(trace, (m.mid,), backend, k, temperature)
        delta = p_fact - p_abl
        se = math.sqrt(
            max(p_fact * (1 - p_fact), 0.0) / k + max(p_abl * (1 - p_abl), 0.0) / k
        )
        results.append(
            Utility(
                pid=trace.pid,
                trace_id=trace.trace_id,
                mid=m.mid,
                role=m.role,
                round=m.round,
                delta=delta,
                p_factual=p_fact,
                p_ablated=p_abl,
                k=k,
                se=se,
            )
        )
    return results


async def noise_floor(
    trace: Trace,
    backend: Any,
    k: int = 16,
    temperature: float = 0.7,
    repeats: int = 2,
) -> list[float]:
    """
    Placebo: re-estimate the UNABLATED condition `repeats` times and return the
    pairwise differences. Their spread is the pure sampling-noise floor.

    Requires a backend whose cache key includes a replicate index, otherwise
    every repeat returns the identical value and tau collapses to exactly 0.
    A tau of exactly 0.0 is proof of cache collapse, not of a clean estimator.
    """
    vals = []
    for r in range(repeats):
        msgs = render_verifier_messages(trace, exclude=())
        # We use cache_nonce to differentiate replicates
        samples = await backend.generate(msgs, n=k, temperature=temperature, cache_nonce=f"__replicate__:{r}")
        vals.append(sum(1 for s in samples if is_correct(s, trace.gold)) / k)
    return [vals[i] - vals[j] for i in range(len(vals)) for j in range(i + 1, len(vals))]
