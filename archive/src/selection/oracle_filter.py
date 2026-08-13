"""
Oracle filter selection strategy.

Selects messages based on oracle information (e.g., ground truth correctness),
serving as an upper-bound baseline for selection strategies.
"""

from typing import Any

from .base import Selector


class OracleFilterSelector(Selector):
    """
    Oracle-based selector for upper-bound evaluation.

    This selector uses ground truth information to select
    only messages that lead to correct final answers,
    serving as an upper bound for selection quality.
    """

    def __init__(self, correct_traces_only: bool = True):
        """
        Initialize the oracle selector.

        Args:
            correct_traces_only: If True, only select from traces with correct answers.
        """
        self.correct_traces_only = correct_traces_only

    def select(
        self,
        traces: list[Any],
        utilities: dict[tuple[str, str], float],
        token_budget: int,
    ) -> dict[str, list[str]]:
        """
        Select messages using oracle information.

        Args:
            traces: List of debate traces.
            utilities: Dictionary mapping (trace_id, mid) to utility scores.
            token_budget: Maximum tokens allowed.

        Returns:
            Dictionary mapping trace_id to list of selected message IDs.
        """
        # Filter traces by correctness if needed
        if self.correct_traces_only:
            traces = [t for t in traces if self._is_correct(t)]

        # Select all messages from correct traces within budget
        selected = {}
        total_tokens = 0

        for trace in traces:
            trace_id = getattr(trace, "trace_id", getattr(trace, "pid", ""))
            for msg in trace.messages:
                tokens = self._estimate_tokens(msg.text)
                if total_tokens + tokens <= token_budget:
                    selected.setdefault(trace_id, []).append(msg.mid)
                    total_tokens += tokens

        return selected

    def _is_correct(self, trace: Any) -> bool:
        """
        Check if a trace leads to the correct answer.

        Args:
            trace: Debate trace to check.

        Returns:
            True if the trace leads to a correct answer.
        """
        return trace.final_correct