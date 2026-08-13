"""
Base selector interface for problem selection.

Provides the abstract base class for all problem selection strategies.
"""

from abc import ABC, abstractmethod
from typing import Any


class Selector(ABC):
    """
    Abstract base class for problem selectors.

    Selectors choose which problems/traces to use for training
    based on various criteria (causal utility, confidence, etc.).
    """

    @abstractmethod
    def select(
        self,
        traces: list[Any],
        utilities: dict[tuple[str, str], float],
        token_budget: int,
    ) -> dict[str, list[str]]:
        """
        Select message IDs for training based on utilities and budget.

        Args:
            traces: List of debate traces.
            utilities: Dictionary mapping (trace_id, mid) to utility scores.
            token_budget: Maximum tokens allowed for selected messages.

        Returns:
            Dictionary mapping trace_id to list of selected message IDs.
        """
        ...

    def _estimate_tokens(self, text: str) -> int:
        """
        Estimate token count for text.

        Rough estimate: ~4 characters per token on average.

        Args:
            text: Text to estimate tokens for.

        Returns:
            Estimated token count.
        """
        return len(text) // 4

    def _compute_selection_score(
        self,
        mid: str,
        utility: float,
        text: str,
        baseline_utility: float = 0.0,
    ) -> float:
        """
        Compute selection score for a message.

        Default implementation: utility * 1.0

        Args:
            mid: Message ID.
            utility: Utility score.
            text: Message text.
            baseline_utility: Baseline utility for comparison.

        Returns:
            Selection score.
        """
        return utility