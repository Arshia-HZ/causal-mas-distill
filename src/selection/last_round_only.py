"""
Last round only selection strategy.

Baseline selector that only selects messages from the last debate round,
simulating selection strategies that only use final answers.
"""

from typing import Any

from .base import Selector


class LastRoundOnlySelector(Selector):
    """
    Selector that only uses messages from the last round.

    This is a baseline that simulates strategies that only
    consider final answers, ignoring intermediate reasoning.
    """

    def select(
        self,
        traces: list[Any],
        utilities: dict[str, float],
        token_budget: int,
    ) -> list[str]:
        """
        Select only last-round messages within token budget.

        Args:
            traces: List of debate traces.
            utilities: Dictionary mapping message IDs to utility scores.
            token_budget: Maximum tokens allowed.

        Returns:
            List of selected message IDs.
        """
        # Find max round across all traces
        max_round = 0
        for trace in traces:
            for msg in trace.messages:
                max_round = max(max_round, msg.round)

        # Collect last-round messages
        last_round_messages = []
        for trace in traces:
            for msg in trace.messages:
                if msg.round == max_round:
                    tokens = self._estimate_tokens(msg.content)
                    utility = utilities.get(msg.mid, 0.0)
                    last_round_messages.append((msg.mid, utility, tokens))

        # Sort by utility (descending) and select within budget
        last_round_messages.sort(key=lambda x: x[1], reverse=True)

        selected = []
        total_tokens = 0

        for mid, utility, tokens in last_round_messages:
            if total_tokens + tokens <= token_budget:
                selected.append(mid)
                total_tokens += tokens

        return selected