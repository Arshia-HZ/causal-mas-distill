"""
Causal utility-based selection.

What changed and why
--------------------
The previous version ranked by raw delta with min_utility=0.0. With your
measured data that selects EVERY message whose delta is >= 0, which is 98.7%
of them, and the ordering inside the huge block of exact ties is just trace
order. The "causal" arm was therefore a first-N-messages arm wearing a
different name, and it would have produced a null result that told you
nothing about causal selection.

Three fixes:

1. Shrinkage instead of thresholding. Rank by the empirical-Bayes posterior
   mean, not the raw estimate. At k<=32 a raw delta is mostly noise; the
   posterior mean pulls unreliable estimates toward the pooled mean so they
   stop winning top-k by luck. Set shrink=False to reproduce the old
   behaviour as an ablation -- that comparison is itself a paper result.

2. Explicit tie policy. Exact ties are the dominant case in a saturated
   regime. Breaking them by trace order silently encodes position. We break
   them randomly under a fixed seed and record n_ties_broken, so the
   confound is visible instead of hidden.

3. Per-trace budget. Filling one global token budget greedily by score lets a
   high-scoring trace consume the whole budget, so different selectors cover
   different numbers of PROBLEMS. That is a coverage confound on top of the
   thing you want to measure. per_trace_budget=True gives every trace the
   same allowance, which makes the arms genuinely comparable.
"""

from __future__ import annotations

import random
from typing import Any

from .base import Selector

try:
    from ..analysis.stats import eb_shrink
except ImportError:  # analysis package optional at import time
    eb_shrink = None


class CausalSelector(Selector):
    """Select messages by interventional utility, with noise-aware ranking."""

    def __init__(
        self,
        min_utility: float | None = None,
        top_k: int | None = None,
        shrink: bool = True,
        seed: int = 0,
        per_trace_budget: bool = True,
        drop_nonpositive: bool = True,
    ):
        """
        Args:
            min_utility: Hard floor on the (shrunken) score. None means no
                floor beyond drop_nonpositive. Avoid using this as the main
                knob -- prefer the budget, which is what the baselines match.
            top_k: Optional cap on number of selected messages.
            shrink: Rank by empirical-Bayes posterior mean rather than raw
                delta. Requires per-message `se`; falls back to raw if absent.
            seed: Seed for random tie-breaking.
            per_trace_budget: Give each trace an equal share of the budget.
            drop_nonpositive: Exclude messages whose score is <= 0. A message
                measured not to help is not training signal.
        """
        self.min_utility = min_utility
        self.top_k = top_k
        self.shrink = shrink
        self.seed = seed
        self.per_trace_budget = per_trace_budget
        self.drop_nonpositive = drop_nonpositive
        self.stats: dict = {}

    def select(
        self,
        traces: list,
        utilities: dict,
        token_budget: int,
        ses: dict | None = None,
    ) -> dict:
        rng = random.Random(self.seed)

        keys, raw, se_list, meta = [], [], [], []
        for trace in traces:
            trace_id = getattr(trace, "trace_id", getattr(trace, "pid", ""))
            for msg in trace.messages:
                key = (trace_id, msg.mid)
                if key not in utilities:
                    continue
                keys.append(key)
                raw.append(float(utilities[key]))
                se_list.append(float((ses or {}).get(key, 0.0)))
                meta.append((trace_id, msg.mid, self._estimate_tokens(msg.text)))

        if not keys:
            self.stats = {"n_candidates": 0}
            return {}

        scores = raw
        diag: dict = {}
        if self.shrink and eb_shrink is not None and any(s > 0 for s in se_list):
            scores, diag = eb_shrink(raw, se_list)

        n_ties = len(scores) - len(set(round(s, 12) for s in scores))

        cands = []
        for (trace_id, mid, tokens), score in zip(meta, scores):
            if self.drop_nonpositive and score <= 0:
                continue
            if self.min_utility is not None and score < self.min_utility:
                continue
            cands.append((trace_id, mid, score, tokens, rng.random()))

        # descending score, random tie-break (never trace order)
        cands.sort(key=lambda x: (-x[2], x[4]))

        if self.top_k is not None:
            cands = cands[: self.top_k]

        selected: dict = {}
        if self.per_trace_budget:
            share = max(token_budget // max(len(traces), 1), 1)
            used: dict = {}
            for trace_id, mid, score, tokens, _ in cands:
                if used.get(trace_id, 0) + tokens <= share:
                    selected.setdefault(trace_id, []).append(mid)
                    used[trace_id] = used.get(trace_id, 0) + tokens
            total = sum(used.values())
        else:
            total = 0
            for trace_id, mid, score, tokens, _ in cands:
                if total + tokens <= token_budget:
                    selected.setdefault(trace_id, []).append(mid)
                    total += tokens

        self.stats = {
            "n_candidates": len(keys),
            "n_after_filter": len(cands),
            "n_selected": sum(len(v) for v in selected.values()),
            "n_traces_covered": len(selected),
            "tokens_used": total,
            "n_ties_broken": n_ties,
            "shrinkage": diag,
        }
        return selected
