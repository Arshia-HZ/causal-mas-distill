"""
Round-robin over several API keys, with per-key concurrency and 429 cooldown.

Drop-in replacement for ApiBackend. The cache key payload is BYTE-IDENTICAL to
ApiBackend's, so every existing cache_probe.jsonl / cache_debates.jsonl entry
still hits. The API key is deliberately NOT part of the cache key.

KEYS ARE NEVER WRITTEN IN CODE OR IN A NOTEBOOK CELL.
Colab:  from google.colab import userdata
        os.environ["GC_API_KEYS"] = userdata.get("GC_API_KEYS")
where the secret GC_API_KEYS is "key1,key2,key3".

USAGE
    from src.backends.multikey import MultiKeyApiBackend
    backend = MultiKeyApiBackend(
        model="deepseek-v3.2",
        base_url="https://api.generalcompute.com/v1",
        api_keys_env="GC_API_KEYS",
        cache_path="/content/drive/MyDrive/cmd/cache_debates_v3.jsonl",
        concurrency_per_key=8,
        max_tokens=1024,
    )
"""

import asyncio
import hashlib
import json
import os
import random
import time
from pathlib import Path

from openai import AsyncOpenAI, BadRequestError

try:
    from openai import RateLimitError
except ImportError:  # older clients
    class RateLimitError(Exception):
        pass


def key_of(payload) -> str:
    """Identical to src.backends.api.key_of. Do not change."""
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()[:32]


def _looks_rate_limited(exc) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    if getattr(exc, "status_code", None) == 429:
        return True
    s = str(exc).lower()
    return "429" in s or "rate limit" in s or "too many requests" in s


class MultiKeyApiBackend:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_keys: list[str] | None = None,
        api_keys_env: str = "GC_API_KEYS",
        cache_path: str = "cache.jsonl",
        concurrency_per_key: int = 8,
        max_tokens: int = 1024,
        supports_n: bool = True,
        extra_body: dict | None = None,
        max_n_per_request: int = 8,
        cooldown_seconds: float = 20.0,
        max_attempts: int = 8,
        # --- drop-in compatibility with src.backends.api.ApiBackend ---
        # These let any existing script switch over with ONE line:
        #     from src.backends.multikey import MultiKeyApiBackend as ApiBackend
        api_key: str | None = None,
        api_key_env: str | None = None,
        concurrency: int | None = None,
    ):
        if concurrency is not None:
            # ApiBackend's `concurrency` was global; here it is per key.
            concurrency_per_key = max(1, int(concurrency))
        keys = list(api_keys or [])
        if not keys and api_key:
            keys = [k.strip() for k in str(api_key).split(",") if k.strip()]
        if not keys and api_key_env:
            raw = os.environ.get(api_key_env, "")
            keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not keys:
            raw = os.environ.get(api_keys_env, "")
            keys = [k.strip() for k in raw.split(",") if k.strip()]
        if not keys:
            i = 1
            while os.environ.get("%s_%d" % (api_keys_env, i)):
                keys.append(os.environ["%s_%d" % (api_keys_env, i)].strip())
                i += 1
        if not keys:
            single = os.environ.get("GC_API_KEY", "").strip()
            if single:
                keys = [single]
        if not keys:
            raise RuntimeError(
                "No API keys. Set %s to a comma-separated list." % api_keys_env)

        self.model = model
        self.max_tokens = max_tokens
        self.supports_n = supports_n
        self.max_n_per_request = max(1, int(max_n_per_request))
        self.extra_body = extra_body or {}
        self.cooldown_seconds = cooldown_seconds
        self.max_attempts = max_attempts

        self.clients = [AsyncOpenAI(api_key=k, base_url=base_url) for k in keys]
        self.sems = [asyncio.Semaphore(concurrency_per_key) for _ in keys]
        self.cool_until = [0.0] * len(keys)
        self.calls = [0] * len(keys)
        self.rate_limited = [0] * len(keys)
        self._rr = 0
        self._rr_lock = asyncio.Lock()
        self.lock = asyncio.Lock()
        self.n_keys = len(keys)
        print("[multikey] %d key(s), %d concurrent each = %d total"
              % (self.n_keys, concurrency_per_key,
                 self.n_keys * concurrency_per_key))

        self.p = Path(cache_path)
        self.p.parent.mkdir(parents=True, exist_ok=True)
        self.cache = {}
        if self.p.exists():
            for line in self.p.open():
                try:
                    r = json.loads(line)
                    self.cache[r["k"]] = r["v"]
                except json.JSONDecodeError:
                    pass
        self.fh = self.p.open("a")

    async def _next_key(self) -> int:
        """Round-robin, skipping keys that are cooling down."""
        for _ in range(3):
            now = time.monotonic()
            async with self._rr_lock:
                for _ in range(self.n_keys):
                    i = self._rr % self.n_keys
                    self._rr += 1
                    if self.cool_until[i] <= now:
                        return i
            wait = max(0.5, min(self.cool_until) - time.monotonic())
            print("[multikey] all keys cooling, sleeping %.1fs" % wait)
            await asyncio.sleep(wait)
        async with self._rr_lock:
            i = self._rr % self.n_keys
            self._rr += 1
        return i

    async def _call(self, client, messages, n, temperature, max_tokens):
        if self.supports_n:
            r = await client.chat.completions.create(
                model=self.model, messages=messages, n=n,
                temperature=temperature, max_completion_tokens=max_tokens,
                extra_body=self.extra_body)
            return [c.message.content for c in r.choices]
        outs = await asyncio.gather(*[
            client.chat.completions.create(
                model=self.model, messages=messages, n=1,
                temperature=temperature, max_completion_tokens=max_tokens,
                extra_body=self.extra_body)
            for _ in range(n)])
        return [o.choices[0].message.content for o in outs]

    async def _one(self, messages, n, temperature, max_tokens):
        last = None
        for attempt in range(self.max_attempts):
            i = await self._next_key()
            try:
                async with self.sems[i]:
                    out = await self._call(self.clients[i], messages, n,
                                           temperature, max_tokens)
                self.calls[i] += 1
                return out
            except BadRequestError:
                raise  # 400 never heals: context overflow, bad params
            except Exception as e:
                last = e
                if _looks_rate_limited(e):
                    self.rate_limited[i] += 1
                    self.cool_until[i] = time.monotonic() + self.cooldown_seconds
                    continue  # immediately try the next key, no backoff
                await asyncio.sleep(min(2 ** attempt, 30) + random.random())
        raise last if last else RuntimeError("exhausted attempts")

    async def generate(self, messages, n=1, temperature=0.7,
                       max_tokens=None, cache_nonce=None) -> list[str]:
        mt = max_tokens or self.max_tokens
        chunks, remaining = [], n
        while remaining > 0:
            take = min(remaining, self.max_n_per_request)
            chunks.append(take)
            remaining -= take

        results = []
        for ci, take in enumerate(chunks):
            nonce = cache_nonce if len(chunks) == 1 else "%s|chunk%d" % (cache_nonce or "", ci)
            key = key_of({"m": messages, "n": take, "t": temperature,
                          "mt": mt, "mo": self.model, "c": nonce})
            if key in self.cache:
                results.extend(self.cache[key])
                continue
            out = await self._one(messages, take, temperature, mt)
            async with self.lock:
                self.cache[key] = out
                self.fh.write(json.dumps({"k": key, "v": out}) + "\n")
                self.fh.flush()
            results.extend(out)
        return results

    def stats(self):
        print("[multikey] calls per key : %s" % self.calls)
        print("[multikey] 429s per key  : %s" % self.rate_limited)
        if self.n_keys > 1 and max(self.calls) > 0:
            spread = min(self.calls) / max(self.calls)
            if spread < 0.5:
                print("[multikey] WARNING: uneven load (%.0f%%). Some keys may "
                      "be invalid or throttled harder." % (100 * spread))

    def close(self):
        self.stats()
        self.fh.close()


def build_role_backends(role_models: dict, base_url: str, cache_path: str,
                        **kw) -> dict:
    """
    Heterogeneous agents: a different model per debate role.

        backends = build_role_backends(
            {"solver": "deepseek-v3.2",
             "critic": "gpt-oss-120b",
             "verifier": "deepseek-v3.2"},
            base_url, "/content/drive/MyDrive/cmd/cache_v3.jsonl")

    One backend per DISTINCT model, shared across roles that use it, so the
    cache and the key pool are shared too.
    """
    made = {}
    out = {}
    for role, model in role_models.items():
        if model not in made:
            made[model] = MultiKeyApiBackend(
                model=model, base_url=base_url, cache_path=cache_path, **kw)
        out[role] = made[model]
    return out
