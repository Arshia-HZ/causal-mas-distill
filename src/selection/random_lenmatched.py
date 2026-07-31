"""
Random length-matched selection strategy.

Baseline selector that randomly selects messages while matching
the length distribution of the causal selector.
"""

import random
from typing import Any

from .base import Selector


class RandomLenMatchedSelector(Selector):
    """
    Random selector with length-matched baseline.

    This is a baseline selector that randomly samples messages
    while maintaining similar length distribution to enable
    fair comparison with causal selection.
    """

    def __init__(self, seed: int = 42):
        """
        Initialize the random selector.

        Args:
            seed: Random seed for reproducibility.
        """
        self.seed = seed
        random.seed(seed)

    def select(
        self,
        traces: list[Any],
        utilities: dict[str, float],
        token_budget: int,
    ) -> list[str]:
        """
        Select messages randomly within token budget.

        Args:
            traces: List of debate traces.
            utilities: Dictionary mapping message IDs to utility scores.
            token_budget: Maximum tokens allowed.

        Returns:
            List of selected message IDs.
        """
        # Collect all messages
        all_messages = []
        for trace in traces:
            for msg in trace.messages:
                tokens = self._estimate_tokens(msg.content)
                all_messages.append((msg.mid, msg.content, tokens))

        # Shuffle
        random.shuffle(all_messages)

        # Select within budget
        selected = []
        total_tokens = 0

        for mid, content, tokens in all_messages:
            if total_tokens + tokens <= token_budget:
                selected.append(mid)
                total_tokens += tokens

        return selected