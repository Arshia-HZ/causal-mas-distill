"""
Process Reward Model (PRM) based selection strategy.

Selects messages based on PRM scores, using a process reward model
to identify high-quality reasoning steps.
"""

from typing import Any

from .base import Selector


class PRMSelector(Selector):
    """
    Selector based on Process Reward Model scores.

    Uses a process reward model to score individual reasoning steps,
    selecting messages with high PRM scores for training.
    """

    def __init__(self, prm_model: Any = None, min_score: float = 0.5):
        """
        Initialize the PRM selector.

        Args:
            prm_model: Process Reward Model for scoring.
            min_score: Minimum PRM score threshold.
        """
        self.prm_model = prm_model
        self.min_score = min_score

    def select(
        self,
        traces: list[Any],
        utilities: dict[tuple[str, str], float],
        token_budget: int,
    ) -> dict[str, list[str]]:
        """
        Select messages based on PRM scores.

        Args:
            traces: List of debate traces.
            utilities: Dictionary mapping (trace_id, mid) to utility scores.
                       For this selector, utilities represent PRM scores.
            token_budget: Maximum tokens allowed.

        Returns:
            Dictionary mapping trace_id to list of selected message IDs.
        """
        # Filter and sort by PRM score
        candidates = []
        for trace in traces:
            trace_id = getattr(trace, "trace_id", getattr(trace, "pid", ""))
            for msg in trace.messages:
                score = utilities.get((trace_id, msg.mid), 0.0)
                if score >= self.min_score:
                    tokens = self._estimate_tokens(msg.text)
                    candidates.append((trace_id, msg.mid, score, tokens))

        # Sort by PRM score descending
        candidates.sort(key=lambda x: x[2], reverse=True)

        # Select within token budget
        selected = {}
        total_tokens = 0

        for trace_id, mid, score, tokens in candidates:
            if total_tokens + tokens <= token_budget:
                selected.setdefault(trace_id, []).append(mid)
                total_tokens += tokens

        return selected