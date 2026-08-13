"""
Random length-matched selection -- the baseline that decides your paper.

What changed and why
--------------------
The previous version shuffled all messages and greedily filled a global token
budget. That is random selection, but it is NOT length-matched, and it is not
count-matched or coverage-matched either. Since the headline claim is
"selecting by utility beats selecting at random at equal budget", any
mismatch here shows up as a fake win for the causal arm. This is the first
thing a reviewer will attack.

This version matches the reference selection on three axes:

1. Per-trace COUNT: same number of messages per trace as the reference.
2. Length STRATUM: each pick is drawn from the same token-length bucket as
   the message it replaces, so the two arms see near-identical token counts.
3. Role composition (optional): preserve the solver/critic mix, which stops
   "random" from accidentally becoming "mostly critics".

Without a reference it falls back to unmatched random and says so in
`self.stats`, so an unmatched run can never be mistaken for a matched one.
"""

from __future__ import annotations

import random
from typing import Any

from .base import Selector


def _bucket(tokens: int) -> int:
    """Coarse log-ish length buckets. Wide enough to always have candidates."""
    if tokens < 128:
        return 0
    if tokens < 256:
        return 1
    if tokens < 512:
        return 2
    if tokens < 1024:
        return 3
    return 4


class RandomLenMatchedSelector(Selector):
    """Random selection matched to a reference selection."""

    def __init__(self, seed: int = 42, match_roles: bool = False):
        self.seed = seed
        self.match_roles = match_roles
        self.stats: dict = {}

    def select(
        self,
        traces: list[Any],
        utilities: dict[tuple[str, str], float],
        token_budget: int,
        reference: dict[str, list[str]] | None = None,
    ) -> dict[str, list[str]]:
        """
        Args:
            reference: trace_id -> selected mids from the arm being matched
                (normally the causal selector). Strongly recommended.
        """
        rng = random.Random(self.seed)

        if reference is None:
            self.stats = {"matched": False, "warning": "no reference: unmatched random"}
            return self._unmatched(traces, token_budget, rng)

        selected: dict[str, list[str]] = {}
        tokens_ref = tokens_new = 0
        n_exact = n_fallback = 0

        for trace in traces:
            trace_id = getattr(trace, "trace_id", getattr(trace, "pid", ""))
            ref_mids = reference.get(trace_id, [])
            if not ref_mids:
                continue

            pool = {}
            for m in trace.messages:
                t = self._estimate_tokens(m.text)
                pool[m.mid] = (t, _bucket(t), m.role)

            taken: set[str] = set()
            for mid in ref_mids:
                if mid not in pool:
                    continue
                want_tok, want_bucket, want_role = pool[mid]
                tokens_ref += want_tok

                cands = [
                    c
                    for c, (t, b, r) in pool.items()
                    if c not in taken
                    and b == want_bucket
                    and (not self.match_roles or r == want_role)
                ]
                if cands:
                    n_exact += 1
                else:
                    # widen: nearest bucket, then anything unused
                    cands = [c for c, (t, b, r) in pool.items() if c not in taken]
                    if not cands:
                        continue
                    cands.sort(key=lambda c: abs(pool[c][1] - want_bucket))
                    cands = cands[: max(1, len(cands) // 2)]
                    n_fallback += 1

                pick = rng.choice(cands)
                taken.add(pick)
                tokens_new += pool[pick][0]

            if taken:
                selected[trace_id] = sorted(taken)

        self.stats = {
            "matched": True,
            "n_selected": sum(len(v) for v in selected.values()),
            "n_traces_covered": len(selected),
            "tokens_reference": tokens_ref,
            "tokens_selected": tokens_new,
            "token_ratio": (tokens_new / tokens_ref) if tokens_ref else 0.0,
            "exact_bucket_matches": n_exact,
            "fallback_matches": n_fallback,
        }
        return selected

    def _unmatched(self, traces, token_budget, rng) -> dict[str, list[str]]:
        all_msgs = []
        for trace in traces:
            trace_id = getattr(trace, "trace_id", getattr(trace, "pid", ""))
            for msg in trace.messages:
                all_msgs.append((trace_id, msg.mid, self._estimate_tokens(msg.text)))
        rng.shuffle(all_msgs)

        selected: dict[str, list[str]] = {}
        total = 0
        for trace_id, mid, tokens in all_msgs:
            if total + tokens <= token_budget:
                selected.setdefault(trace_id, []).append(mid)
                total += tokens
        return selected
