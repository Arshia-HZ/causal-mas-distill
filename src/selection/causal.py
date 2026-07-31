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
        utilities: dict[str, float],
        token_budget: int,
    ) -> list[str]:
        """
        Select messages based on causal utility.

        Args:
            traces: List of debate traces.
            utilities: Dictionary mapping message IDs to utility scores.
            token_budget: Maximum tokens allowed.

        Returns:
            List of selected message IDs.
        """
        # Filter and sort by utility
        candidates = []
        for trace in traces:
            for msg in trace.messages:
                utility = utilities.get(msg.mid, 0.0)
                if utility >= self.min_utility:
                    tokens = self._estimate_tokens(msg.content)
                    score = self._compute_selection_score(msg.mid, utility, msg.content)
                    candidates.append((msg.mid, score, tokens))

        # Sort by score descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Apply top_k if set
        if self.top_k is not None:
            candidates = candidates[: self.top_k]

        # Select within token budget (greedy)
        selected = []
        total_tokens = 0

        for mid, score, tokens in candidates:
            if total_tokens + tokens <= token_budget:
                selected.append(mid)
                total_tokens += tokens

        return selected