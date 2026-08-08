"""
Last-round-only selection.

What changed and why
--------------------
The previous version computed a single GLOBAL max round across all traces and
kept only messages at that round. Two consequences:

1. The debate harness appends a verifier at round max_rounds+1, so the global
   max is the verifier round. The baseline was selecting only verifier
   messages -- i.e. the final answer restated -- not the last DEBATE round.
2. Any trace that terminated early (the harness breaks when a generation
   comes back empty) has a lower max round and contributed ZERO messages,
   silently shrinking this arm's coverage relative to every other arm.

Both are fixed: the max round is computed per trace, and the verifier is
excluded by default so this measures "train on the final revision only",
which is the baseline you actually want to beat.
"""

from __future__ import annotations

from typing import Any

from .base import Selector


class LastRoundOnlySelector(Selector):
    """Keep only the final round of each trace."""

    def __init__(self, include_verifier: bool = False, per_trace_budget: bool = True):
        self.include_verifier = include_verifier
        self.per_trace_budget = per_trace_budget
        self.stats: dict = {}

    def select(
        self,
        traces: list[Any],
        utilities: dict[tuple[str, str], float],
        token_budget: int,
    ) -> dict[str, list[str]]:
        selected: dict[str, list[str]] = {}
        n_traces = max(len(traces), 1)
        share = max(token_budget // n_traces, 1)
        total = 0
        n_empty = 0

        for trace in traces:
            trace_id = getattr(trace, "trace_id", getattr(trace, "pid", ""))
            msgs = [
                m
                for m in trace.messages
                if self.include_verifier or m.role != "verifier"
            ]
            if not msgs:
                n_empty += 1
                continue

            local_max = max(m.round for m in msgs)
            picks = [m for m in msgs if m.round == local_max]

            used = 0
            for m in picks:
                tokens = self._estimate_tokens(m.text)
                cap = share if self.per_trace_budget else token_budget
                current = used if self.per_trace_budget else total
                if current + tokens <= cap:
                    selected.setdefault(trace_id, []).append(m.mid)
                    used += tokens
                    total += tokens

            if trace_id not in selected:
                n_empty += 1

        self.stats = {
            "n_selected": sum(len(v) for v in selected.values()),
            "n_traces_covered": len(selected),
            "n_traces_empty": n_empty,
            "tokens_used": total,
        }
        return selected
