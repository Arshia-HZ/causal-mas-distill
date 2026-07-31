"""
Template renderer for formatting training data.

Single template for ALL selectors - provides format control
for converting debate traces into training examples.
"""

from typing import Any


class TemplateRenderer:
    """
    Renders debate traces into training templates.

    Uses a single unified template format that works for
    all selection strategies, ensuring consistency.
    """

    SYSTEM_PROMPT = """You are a helpful AI assistant that engages in critical thinking exercises.
When solving problems, you provide step-by-step reasoning and are open to feedback."""

    USER_TEMPLATE = "{problem}"

    ASSISTANT_TEMPLATE = "{solution}"

    @classmethod
    def render_trace(
        cls,
        problem: str,
        solution: str,
        critique: str | None = None,
        revision: str | None = None,
    ) -> dict[str, str]:
        """
        Render a debate trace into a training example.

        Args:
            problem: The original problem statement.
            solution: The solution text.
            critique: Optional critique text.
            revision: Optional revision text.

        Returns:
            Dictionary with 'prompt' and 'completion' keys.
        """
        # Build the conversation
        messages = []

        # System message
        messages.append({"role": "system", "content": cls.SYSTEM_PROMPT})

        # User message with problem
        messages.append({"role": "user", "content": cls.USER_TEMPLATE.format(problem=problem)})

        # Assistant response
        messages.append({"role": "assistant", "content": cls.ASSISTANT_TEMPLATE.format(solution=solution)})

        # Format as prompt
        prompt = cls._format_messages(messages[:-1])  # Exclude completion
        completion = messages[-1]["content"]

        return {
            "prompt": prompt,
            "completion": completion,
            "messages": messages,
        }

    @classmethod
    def render_messages(
        cls,
        problem: str,
        messages: list[dict[str, Any]],
    ) -> dict[str, str]:
        """
        Render a list of messages into a training example.

        Args:
            problem: The original problem statement.
            messages: List of message dicts from a trace.

        Returns:
            Dictionary with 'prompt' and 'completion' keys.
        """
        # Build conversation
        conversation = [{"role": "system", "content": cls.SYSTEM_PROMPT}]
        conversation.append({"role": "user", "content": cls.USER_TEMPLATE.format(problem=problem)})

        # Add all assistant messages
        for msg in messages:
            if msg.get("role") in ("assistant", "revision"):
                conversation.append({
                    "role": "assistant",
                    "content": msg.get("content", ""),
                })

        # Last assistant message is the completion
        prompt = cls._format_messages(conversation[:-1])
        completion = conversation[-1]["content"]

        return {
            "prompt": prompt,
            "completion": completion,
            "messages": conversation,
        }

    @classmethod
    def _format_messages(cls, messages: list[dict[str, str]]) -> str:
        """
        Format messages into a prompt string.

        Args:
            messages: List of message dicts.

        Returns:
            Formatted prompt string.
        """
        result = ""
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                result += f"System: {content}\n\n"
            elif role == "user":
                result += f"User: {content}\n\n"
            elif role == "assistant":
                result += f"Assistant: {content}\n\n"
        result += "Assistant:"
        return result

    @classmethod
    def render_for_sft(
        cls,
        traces: list[Any],
        selected_mids: list[str],
    ) -> list[dict[str, str]]:
        """
        Render selected messages from traces for SFT training.

        Args:
            traces: List of debate traces.
            selected_mids: List of selected message IDs.

        Returns:
            List of training examples.
        """
        # Build lookup for selected messages
        mid_to_trace = {}
        for trace in traces:
            for msg in trace.messages:
                if msg.mid in selected_mids:
                    mid_to_trace[msg.mid] = trace

        # Render each selected message
        examples = []
        for mid in selected_mids:
            trace = mid_to_trace.get(mid)
            if trace is None:
                continue

            # Find the message
            msg = None
            for m in trace.messages:
                if m.mid == mid:
                    msg = m
                    break

            if msg is None:
                continue

            # Get context (previous messages)
            context = []
            for m in trace.messages:
                if m.round < msg.round or (m.round == msg.round and m.created_at < msg.created_at):
                    context.append(m)

            rendered = cls.render_messages(trace.problem, context + [msg])
            examples.append(rendered)

        return examples