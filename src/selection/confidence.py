"""
Confidence-based selection strategy.

Selects messages based on confidence scores (e.g., from a PRM or
model uncertainty estimates).
"""

from typing import Any

from .base import Selector


class ConfidenceSelector(Selector):
    """
    Selector based on response confidence.

    Selects messages that the model or PRM scores as high confidence,
    serving as an alternative to causal selection.
    """

    def __init__(self, min_confidence: float = 0.7, top_k: int | None = None):
        """
        Initialize the confidence selector.

        Args:
            min_confidence: Minimum confidence threshold.
            top_k: Optional limit on number of selections.
        """
        self.min_confidence = min_confidence
        self.top_k = top_k

    def select(
        self,
        traces: list[Any],
        utilities: dict[str, float],
        token_budget: int,
    ) -> list[str]:
        """
        Select messages based on confidence scores.

        Args:
            traces: List of debate traces.
            utilities: Dictionary mapping message IDs to utility scores.
                       For this selector, utilities represent confidence.
            token_budget: Maximum tokens allowed.

        Returns:
            List of selected message IDs.
        """
        # Filter and sort by confidence (stored as utility)
        candidates = []
        for trace in traces:
            for msg in trace.messages:
                confidence = utilities.get(msg.mid, 0.0)
                if confidence >= self.min_confidence:
                    tokens = self._estimate_tokens(msg.content)
                    candidates.append((msg.mid, confidence, tokens))

        # Sort by confidence descending
        candidates.sort(key=lambda x: x[1], reverse=True)

        # Apply top_k if set
        if self.top_k is not None:
            candidates = candidates[: self.top_k]

        # Select within token budget
        selected = []
        total_tokens = 0

        for mid, confidence, tokens in candidates:
            if total_tokens + tokens <= token_budget:
                selected.append(mid)
                total_tokens += tokens

        return selected