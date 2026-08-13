"""
API Backend for OpenAI-compatible model inference.

Works with DeepSeek, OpenRouter, Gemini-compatible, DashScope.
Includes disk caching, retry logic, and semaphore-based concurrency control.
"""

import asyncio
import hashlib
import json
import os
import random
from pathlib import Path

from openai import AsyncOpenAI, BadRequestError


def key_of(payload) -> str:
    """Generate cache key from payload using SHA256 hash."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:32]


class ApiBackend:
    """
    OpenAI-compatible async backend with caching and retry.
    
    Works with DeepSeek, OpenRouter, Gemini-compatible, DashScope.
    Supports n>1 sampling via batching for APIs that don't support n parameter.
    """

    def __init__(
        self,
        model: str,
        base_url: str,
        api_key_env: str = None,
        api_key: str = None,
        cache_path: str = "cache.jsonl",
        concurrency: int = 24,
        max_tokens: int = 768,
        supports_n: bool = True,
        extra_body: dict | None = None,
        max_n_per_request: int = 8,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.supports_n = supports_n
        # Providers cap `n` per request (yours caps it at 8). That is a
        # PER-REQUEST cap, not a cap on k. generate() splits large k across
        # several requests with distinct cache keys, so k=32/64 is available.
        self.max_n_per_request = max(1, int(max_n_per_request))
        self.extra_body = extra_body or {}
        
        actual_key = api_key or (os.environ[api_key_env] if api_key_env else os.environ.get("OPENAI_API_KEY", ""))
        
        self.client = AsyncOpenAI(
            api_key=actual_key,
            base_url=base_url
        )
        
        self.sem = asyncio.Semaphore(concurrency)
        self.lock = asyncio.Lock()
        
        self.p = Path(cache_path)
        self.p.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing cache
        self.cache = {}
        if self.p.exists():
            for line in self.p.open():
                try:
                    r = json.loads(line)
                    self.cache[r["k"]] = r["v"]
                except json.JSONDecodeError:
                    pass  # tolerate truncated last line
        
        self.fh = self.p.open("a")

    async def _one(self, messages, n: int, temperature: float, max_tokens: int):
        """Single generation attempt with retry logic."""
        for attempt in range(6):
            try:
                if self.supports_n:
                    r = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        n=n,
                        temperature=temperature,
                        max_completion_tokens=max_tokens,
                        extra_body=self.extra_body,
                    )
                    return [c.message.content for c in r.choices]
                
                # Batch requests for APIs without n support
                outs = await asyncio.gather(*[
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        n=1,
                        temperature=temperature,
                        max_completion_tokens=max_tokens,
                        extra_body=self.extra_body,
                    )
                    for _ in range(n)
                ])
                return [o.choices[0].message.content for o in outs]
                
            except BadRequestError:
                raise  # 400s (context overflow, bad params) never heal by retrying
            except Exception:
                if attempt == 5:
                    raise
                await asyncio.sleep(2 ** attempt + random.random())

    async def generate(
        self,
        messages: list[dict],
        n: int = 1,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        cache_nonce: str | None = None,
    ) -> list[str]:
        """
        Generate completions with caching.
        
        Args:
            messages: Chat messages.
            n: Number of samples.
            temperature: Sampling temperature.
            max_tokens: Maximum generated tokens.
            cache_nonce: Optional string to differentiate cache keys.
            
        Returns:
            List of generated text completions.
        """
        mt = max_tokens or self.max_tokens

        # Split n into provider-sized chunks. Each chunk gets its own cache
        # key (chunk index is part of the key) so replicates never collapse
        # onto one cached draw, which would silently zero the variance.
        chunks = []
        remaining = n
        while remaining > 0:
            take = min(remaining, self.max_n_per_request)
            chunks.append(take)
            remaining -= take

        results: list[str] = []
        for ci, take in enumerate(chunks):
            nonce = cache_nonce if len(chunks) == 1 else f"{cache_nonce or ''}|chunk{ci}"
            key = key_of({"m": messages, "n": take, "t": temperature,
                          "mt": mt, "mo": self.model, "c": nonce})

            if key in self.cache:
                results.extend(self.cache[key])
                continue

            async with self.sem:
                out = await self._one(messages, take, temperature, mt)

            async with self.lock:
                self.cache[key] = out
                self.fh.write(json.dumps({"k": key, "v": out}) + "\n")
                self.fh.flush()

            results.extend(out)

        return results

    def close(self):
        """Close file handle."""
        self.fh.close()