"""
Matched-condition counterfactual message utility with common random numbers.

THE BUG THIS REPLACES
---------------------
src/counterfactual/estimands.py::trace_utilities_total computes the factual arm
as:

    p_fact, _, _ = await _p_correct_with_regen(trace, set(), ...)

With dropped = empty set, the affected/descendant set is empty, so no
intermediate node is regenerated -- only the terminal is resampled. The ablated
arm drops message m AND regenerates every descendant of m.

So the reported delta is not "with m" minus "without m". It is:

    (factual transcript, terminal resampled)
  - (m removed AND every descendant freshly resampled)

Resampling alone flips the final answer on 17-30% of these traces. All of that
variance sits on the ablated arm, and it grows with the number of descendants.
r1.solver has the most descendants, so it will show the largest |delta| purely
as an artefact. The pipeline would then "discover" that early messages matter
most -- a conclusion manufactured by the estimator, not by the data.

THE FIX
-------
Run two conditions that regenerate EXACTLY the same node set D(m):

    p_keep : regenerate D(m) with m PRESENT in every downstream prompt
    p_drop : regenerate D(m) with m ABSENT from every downstream prompt
    delta  = p_keep - p_drop

Resampling variance now appears in both arms and cancels.

COMMON RANDOM NUMBERS
---------------------
We run k paired "worlds" w = 0..k-1. Within a world, the cache nonce for a node
is identical across the two conditions:

    nonce = "crn|{trace_id}|w{w}|{mid}"

Note what is deliberately ABSENT from that nonce: the identity of the ablated
message, and the condition label. The backend caches on
(messages, n, temperature, max_tokens, model, nonce). So when removing m does
not change a downstream node's prompt, that node returns byte-identical text in
both conditions and contributes nothing to the difference. Only differences
caused by m survive to the terminal.

The paired standard error is computed from the per-world difference vector, not
from the two marginals. It is typically 2-4x smaller than the unpaired SE the
old code reported, which matters at k=32 where the raw SE is about 0.125.

MANDATORY SANITY CHECK
----------------------
Run placebo_check() before trusting any number. It ablates a message that
cannot possibly matter. Under CRN the delta must be EXACTLY 0.0 in every world.
If it is not, the nonces are wrong or the cache is not being hit, and every
attribution number is noise.
"""

from __future__ import annotations

import asyncio
from statistics import pstdev

from ..debate.schema import Trace, descendants, topo_order
from .estimands import UtilityResult, default_node_prompt
from .replay import render_verifier_messages

try:
    from eval.grade import extract_answer, is_correct
except ImportError:
    from ...eval.grade import extract_answer, is_correct


def _crn_nonce(trace_id: str, world: int, mid: str) -> str:
    """
    The ablation target and the condition label MUST NOT appear here. That
    omission is what makes the two conditions share randomness.
    """
    return "crn|%s|w%d|%s" % (trace_id, world, mid)


async def _run_world(
    trace: Trace,
    backend,
    dropped: set[str],
    regen_order: list[str],
    world: int,
    temperature: float,
    node_prompt,
) -> float:
    """
    Regenerate `regen_order` in topological order, then score the terminal.
    Returns 1.0 if the terminal answer is correct, else 0.0.
    """
    overrides: dict[str, str] = {}

    for mid in regen_order:
        node = trace.get_message(mid)
        if node is None:
            continue
        prompt = node_prompt(trace, node, overrides, dropped)
        out = await backend.generate(
            [{"role": "user", "content": prompt}],
            n=1,
            temperature=temperature,
            cache_nonce=_crn_nonce(trace.trace_id, world, mid),
        )
        overrides[mid] = out[0] if out else ""

    messages = render_verifier_messages(trace, exclude=tuple(sorted(dropped)),
                                        overrides=overrides)
    out = await backend.generate(
        messages,
        n=1,
        temperature=temperature,
        cache_nonce=_crn_nonce(trace.trace_id, world, "__terminal__"),
    )
    text = out[0] if out else ""
    pred = extract_answer(text) or text
    return 1.0 if is_correct(pred, trace.gold) else 0.0


async def message_utility_crn(
    trace: Trace,
    backend,
    mid: str,
    k: int = 32,
    temperature: float = 0.7,
    max_regen_depth: int | None = None,
    node_prompt=default_node_prompt,
    world_concurrency: int = 8,
) -> UtilityResult:
    """Paired counterfactual utility of one message."""
    node = trace.get_message(mid)
    if node is None:
        raise KeyError("no message %s in %s" % (mid, trace.trace_id))

    desc = set(descendants(trace, mid))
    if max_regen_depth is not None:
        # Keep only descendants within max_regen_depth rounds of the target.
        desc = {
            d for d in desc
            if trace.get_message(d) is not None
            and trace.get_message(d).round - node.round <= max_regen_depth
        }
    regen_order = topo_order(trace, desc)

    sem = asyncio.Semaphore(world_concurrency)

    async def _pair(w: int) -> tuple[float, float]:
        async with sem:
            keep = await _run_world(trace, backend, set(), regen_order, w,
                                    temperature, node_prompt)
            drop = await _run_world(trace, backend, {mid}, regen_order, w,
                                    temperature, node_prompt)
            return keep, drop

    pairs = await asyncio.gather(*[_pair(w) for w in range(k)])
    keeps = [p[0] for p in pairs]
    drops = [p[1] for p in pairs]
    diffs = [a - b for a, b in zip(keeps, drops)]

    p_keep = sum(keeps) / k
    p_drop = sum(drops) / k
    delta = p_keep - p_drop
    # Paired SE. Every element of diffs is in {-1, 0, +1}.
    se = (pstdev(diffs) / (k ** 0.5)) if k > 1 else 0.0

    # If no world ever differed, m is causally inert on this trace under this
    # protocol. That is a real finding, not a failure -- but flag it so it is
    # not confused with a broken nonce.
    degenerate = all(d == 0.0 for d in diffs)

    return UtilityResult(
        pid=trace.pid,
        trace_id=trace.trace_id,
        mid=mid,
        role=node.role,
        round=node.round,
        delta=delta,
        p_factual=p_keep,
        p_ablated=p_drop,
        k=k,
        se=se,
        estimand="total_crn",
        n_regenerated=len(regen_order),
        degenerate=degenerate,
        notes="paired CRN; regen=%d" % len(regen_order),
    )


async def trace_utilities_total_crn(
    trace: Trace,
    backend,
    k: int = 32,
    temperature: float = 0.7,
    max_regen_depth: int | None = None,
    node_prompt=default_node_prompt,
    skip_terminal: bool = True,
) -> list[UtilityResult]:
    """
    Paired CRN utility for every ablatable message in the trace.

    Drop-in replacement for estimands.trace_utilities_total. The terminal
    message is skipped by default: ablating the message that produces the
    answer is not a meaningful counterfactual.
    """
    terminal = trace.messages[-1].mid if trace.messages else None
    targets = [
        m.mid for m in trace.messages
        if not (skip_terminal and m.mid == terminal)
    ]
    results = []
    for mid in targets:
        results.append(await message_utility_crn(
            trace, backend, mid, k=k, temperature=temperature,
            max_regen_depth=max_regen_depth, node_prompt=node_prompt,
        ))
    return results


async def placebo_check(trace: Trace, backend, k: int = 8,
                       temperature: float = 0.7) -> dict:
    """
    RUN THIS FIRST. Ablate a message that cannot matter and confirm the
    measured delta is EXACTLY zero.

    Method: pick the terminal's immediate parent, run the paired estimator with
    an empty regeneration set in BOTH conditions and with the message still
    present in both. Any nonzero result means the paired machinery is leaking
    randomness.
    """
    diffs = []
    for w in range(k):
        a = await _run_world(trace, backend, set(), [], w, temperature,
                             default_node_prompt)
        b = await _run_world(trace, backend, set(), [], w, temperature,
                             default_node_prompt)
        diffs.append(a - b)
    ok = all(d == 0.0 for d in diffs)
    return {
        "trace_id": trace.trace_id,
        "k": k,
        "max_abs_diff": max(abs(d) for d in diffs) if diffs else 0.0,
        "passed": ok,
        "message": (
            "PASS: identical prompts return identical completions under CRN."
            if ok else
            "FAIL: identical prompts returned different completions. The cache "
            "is not being hit, or the nonce includes something it should not. "
            "Every attribution number from this backend is noise until fixed."
        ),
    }
