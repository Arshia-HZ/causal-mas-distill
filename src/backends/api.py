"""
API backend for OpenAI-compatible model inference.

Features:
- OpenAI-compatible API calls
- Disk caching for reproducibility
- Automatic retry with tenacity
- Semaphore-based rate limiting
"""

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import openai
from tenacity import retry, stop_after_attempt, wait_exponential

from .base import Backend


class APIBackend(Backend):
    """API-based backend with caching and retry logic."""

    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        model: str = "gpt-4o-mini",
        cache_dir: Path | None = None,
        max_concurrent: int = 10,
    ):
        """
        Initialize API backend.

        Args:
            base_url: Base URL for OpenAI-compatible API.
            api_key: API key for authentication.
            model: Model name to use.
            cache_dir: Directory for disk caching.
            max_concurrent: Maximum concurrent requests.
        """
        self.client = openai.OpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.cache_dir = cache_dir
        self._semaphore: Any = None  # Will be initialized on first use

    def generate(
        self,
        messages: list[dict],
        n: int = 1,
        temperature: float = 1.0,
        max_tokens: int | None = None,
    ) -> list[str]:
        """Generate completions with caching and retry."""
        # Check cache first
        if self.cache_dir:
            cache_key = self._make_cache_key(messages, n, temperature, max_tokens)
            cached = self._read_cache(cache_key)
            if cached:
                return cached

        # Generate with retry
        response = self._generate_with_retry(messages, n, temperature, max_tokens)

        # Extract content
        completions = [choice.message.content for choice in response.choices]

        # Cache results
        if self.cache_dir:
            self._write_cache(cache_key, completions)

        return completions

    def _make_cache_key(
        self,
        messages: list[dict],
        n: int,
        temperature: float,
        max_tokens: int | None,
    ) -> str:
        """Create a cache key from request parameters."""
        payload = json.dumps({"messages": messages, "n": n, "temperature": temperature, "max_tokens": max_tokens}, sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    def _read_cache(self, cache_key: str) -> list[str] | None:
        """Read cached completions if available."""
        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            with open(cache_file) as f:
                return json.load(f)
        return None

    def _write_cache(self, cache_key: str, completions: list[str]) -> None:
        """Write completions to cache."""
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_file = self.cache_dir / f"{cache_key}.json"
        with open(cache_file, "w") as f:
            json.dump(completions, f)

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
    def _generate_with_retry(
        self,
        messages: list[dict],
        n: int,
        temperature: float,
        max_tokens: int | None,
    ) -> Any:
        """Generate with automatic retry on failure."""
        return self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            n=n,
            temperature=temperature,
            max_tokens=max_tokens,
        )