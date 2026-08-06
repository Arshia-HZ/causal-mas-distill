"""
Causal utility-based selection strategy.

Selects messages based on estimated causal utility (Δ),
focusing on messages with the highest causal impact.
"""

from typing import Any

from .base import Selector


class CausalSelector(Selector):
    """
    Selector based on causal utility estimation.

    This is the primary selector for the thesis, selecting
    messages that have the highest estimated causal effect
    on downstream outcomes.
    """

    def __init__(self, min_utility: float = 0.0, top_k: int | None = None):
        """
        Initialize the causal selector.

        Args:
            min_utility: Minimum utility threshold for selection.
            top_k: Optional limit on number of selections.
        """
        self.min_utility = min_utility
        self.top_k = top_k

    def select(
        self,
        traces: list[Any],
        utilities: dict[tuple[str, str], float],
        token_budget: int,
    ) -> dict[str, list[str]]:
        """
        Select messages based on causal utility.

        Args:
            traces: List of debate traces.
            utilities: Dictionary mapping (trace_id, mid) to utility scores.
            token_budget: Maximum tokens allowed.

        Returns:
            Dictionary mapping trace_id to list of selected message IDs.
        """
        # Filter and sort by utility
        candidates = []
        for trace in traces:
            trace_id = getattr(trace, "trace_id", getattr(trace, "pid", ""))
            for msg in trace.messages:
                utility = utilities.get((trace_id, msg.mid), 0.0)
                if utility >= self.min_utility:
                    tokens = self._estimate_tokens(msg.text)
                    score = self._compute_selection_score(msg.mid, utility, msg.text)
                    candidates.append((trace_id, msg.mid, score, tokens))

        # Sort by score descending
        candidates.sort(key=lambda x: x[2], reverse=True)

        # Apply top_k if set
        if self.top_k is not None:
            candidates = candidates[: self.top_k]

        # Select within token budget (greedy)
        selected = {}
        total_tokens = 0

        for trace_id, mid, score, tokens in candidates:
            if total_tokens + tokens <= token_budget:
                selected.setdefault(trace_id, []).append(mid)
                total_tokens += tokens

        return selected