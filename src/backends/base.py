"""
Base backend interface for model inference.

Provides a unified interface for generating completions from different backends
(API-based or local vLLM) with consistent behavior.
"""

from abc import ABC, abstractmethod
from typing import Protocol


class Backend(Protocol):
    """Protocol defining the backend interface."""

    @abstractmethod
    async def generate(
        self,
        messages: list[dict],
        n: int = 1,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        cache_nonce: str | None = None,
    ) -> list[str]:
        """
        Generate completions for the given messages.

        Args:
            messages: List of message dicts with 'role' and 'content' keys.
            n: Number of completions to generate.
            temperature: Sampling temperature.
            max_tokens: Maximum tokens to generate.

        Returns:
            List of generated completion strings.
        """
        ...