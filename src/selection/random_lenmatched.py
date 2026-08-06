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
        self._rng = random.Random(seed)

    def select(
        self,
        traces: list[Any],
        utilities: dict[tuple[str, str], float],
        token_budget: int,
    ) -> dict[str, list[str]]:
        """
        Select messages randomly within token budget.

        Args:
            traces: List of debate traces.
            utilities: Dictionary mapping (trace_id, mid) to utility scores.
            token_budget: Maximum tokens allowed.

        Returns:
            Dictionary mapping trace_id to list of selected message IDs.
        """
        # Collect all messages
        all_messages = []
        for trace in traces:
            trace_id = getattr(trace, "trace_id", getattr(trace, "pid", ""))
            for msg in trace.messages:
                tokens = self._estimate_tokens(msg.text)
                all_messages.append((trace_id, msg.mid, tokens))

        # Shuffle
        self._rng.shuffle(all_messages)

        # Select within budget
        selected = {}
        total_tokens = 0

        for trace_id, mid, tokens in all_messages:
            if total_tokens + tokens <= token_budget:
                selected.setdefault(trace_id, []).append(mid)
                total_tokens += tokens

        return selected