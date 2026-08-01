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

from openai import AsyncOpenAI


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
        api_key_env: str,
        cache_path: str,
        concurrency: int = 24,
        max_tokens: int = 768,
        supports_n: bool = True,
    ):
        self.model = model
        self.max_tokens = max_tokens
        self.supports_n = supports_n
        
        self.client = AsyncOpenAI(
            api_key=os.environ[api_key_env],
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

    async def _one(self, messages, n: int, temperature: float):
        """Single generation attempt with retry logic."""
        for attempt in range(6):
            try:
                if self.supports_n:
                    r = await self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        n=n,
                        temperature=temperature,
                        max_tokens=self.max_tokens
                    )
                    return [c.message.content for c in r.choices]
                
                # Batch requests for APIs without n support
                outs = await asyncio.gather(*[
                    self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        n=1,
                        temperature=temperature,
                        max_tokens=self.max_tokens
                    )
                    for _ in range(n)
                ])
                return [o.choices[0].message.content for o in outs]
                
            except Exception:
                if attempt == 5:
                    raise
                await asyncio.sleep(2 ** attempt + random.random())

    async def generate(
        self,
        messages: list[dict],
        n: int = 1,
        temperature: float = 0.7
    ) -> list[str]:
        """
        Generate completions with caching.
        
        Args:
            messages: Chat messages.
            n: Number of samples.
            temperature: Sampling temperature.
            
        Returns:
            List of generated text completions.
        """
        k = key_of({"m": messages, "n": n, "t": temperature, "mo": self.model})
        
        if k in self.cache:
            return self.cache[k]
        
        async with self.sem:
            out = await self._one(messages, n, temperature)
        
        async with self.lock:
            self.cache[k] = out
            self.fh.write(json.dumps({"k": k, "v": out}) + "\n")
            self.fh.flush()
        
        return out

    def close(self):
        """Close file handle."""
        self.fh.close()