"""
Counterfactual replay module.

Provides topological downstream regeneration of traces,
enabling counterfactual analysis by replaying parts of the debate
with modified inputs.
"""

from collections import deque
from typing import Any

from ..backends.base import Backend
from ..debate.schema import Message, MessageRole, RoundType, Trace


class CounterfactualReplay:
    """
    Replayer for counterfactual analysis of debate traces.

    This enables regeneration of downstream messages when upstream
    messages are modified, using topological ordering.
    """

    def __init__(self, backend: Backend, max_tokens: int | None = None, temperature: float = 0.7):
        """
        Initialize the counterfactual replayer.

        Args:
            backend: Backend for model inference.
            max_tokens: Maximum tokens per generation.
            temperature: Sampling temperature.
        """
        self.backend = backend
        self.max_tokens = max_tokens
        self.temperature = temperature

    def replay_from(
        self,
        trace: Trace,
        from_mid: str,
        modified_content: str | None = None,
    ) -> Trace:
        """
        Replay the trace starting from a modified message.

        Args:
            trace: Original trace.
            from_mid: Message ID to start replaying from.
            modified_content: New content for the starting message (if any).

        Returns:
            New trace with regenerated downstream messages.
        """
        # Build the DAG
        dag = self._build_dag(trace)

        # Find all downstream messages
        downstream = self._get_downstream(dag, from_mid)

        # Create new trace with prefix
        new_trace = self._create_prefix_trace(trace, from_mid, modified_content)

        # Process downstream messages in topological order
        for mid in downstream:
            msg = dag[mid]
            parent_msg = dag.get(msg.parent_id) if msg.parent_id else None

            # Get parent content for context
            if parent_msg and parent_msg.parent_id:
                grandparent_msg = dag.get(parent_msg.parent_id)
            else:
                grandparent_msg = None

            # Regenerate the message
            new_content = self._regenerate_message(msg, parent_msg, grandparent_msg, new_trace.problem)

            # Add to new trace
            new_trace.add_message(
                content=new_content,
                role=msg.role,
                round_type=msg.round_type,
                parent_id=new_trace.messages[-1].mid if new_trace.messages else None,
                metadata=msg.metadata,
            )

        return new_trace

    def _build_dag(self, trace: Trace) -> dict[str, Message]:
        """Build a DAG (directed acyclic graph) from the trace."""
        return {msg.mid: msg for msg in trace.messages}

    def _get_downstream(self, dag: dict[str, Message], from_mid: str) -> list[str]:
        """
        Get all message IDs downstream of the given message, in topological order.

        Args:
            dag: Dictionary mapping message IDs to messages.
            from_mid: Starting message ID.

        Returns:
            List of downstream message IDs in topological order.
        """
        visited = set()
        result = []

        def topological_sort(start_mid: str) -> None:
            """DFS topological sort."""
            if start_mid in visited:
                return
            visited.add(start_mid)

            # Find children
            children = [mid for mid, msg in dag.items() if msg.parent_id == start_mid]
            for child_mid in children:
                topological_sort(child_mid)
                if child_mid not in result:
                    result.append(child_mid)

        topological_sort(from_mid)
        return result

    def _create_prefix_trace(
        self,
        trace: Trace,
        from_mid: str,
        modified_content: str | None,
    ) -> Trace:
        """Create a new trace with the prefix up to (but not including) from_mid."""
        new_trace = Trace(problem=trace.problem)

        for msg in trace.messages:
            if msg.mid == from_mid:
                break
            content = modified_content if msg.mid == from_mid and modified_content else msg.content
            new_trace.add_message(
                content=content,
                role=msg.role,
                round_type=msg.round_type,
                parent_id=msg.parent_id,
                metadata=msg.metadata,
            )

        return new_trace

    def _regenerate_message(
        self,
        msg: Message,
        parent_msg: Message | None,
        grandparent_msg: Message | None,
        problem: str,
    ) -> str:
        """
        Regenerate a message based on its type and context.

        Args:
            msg: Message to regenerate.
            parent_msg: Parent message.
            grandparent_msg: Grandparent message (for context).
            problem: Original problem statement.

        Returns:
            Generated message content.
        """
        from ..debate.prompts import get_critique_prompt, get_revision_prompt

        if msg.round_type == RoundType.CRITIQUE and parent_msg:
            prompt = get_critique_prompt(parent_msg.content)
        elif msg.round_type == RoundType.REVISION and parent_msg and grandparent_msg:
            prompt = get_revision_prompt(problem, grandparent_msg.content, parent_msg.content)
        else:
            # Fallback: just return original content
            return msg.content

        messages = [{"role": "user", "content": prompt}]
        results = self.backend.generate(messages, n=1, temperature=self.temperature, max_tokens=self.max_tokens)

        return results[0] if results else msg.content