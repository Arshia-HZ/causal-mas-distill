"""
Free deterministic mock backend for $0 pipeline validation.

The entire counterfactual pipeline can and should be validated against this
backend BEFORE any paid API call. It encodes a known synthetic causal
structure, so the estimator can be checked for correctness of sign, scale,
and noise behaviour against ground truth.

Contract match with ApiBackend:
    await backend.generate(messages, n=..., temperature=...) -> list[str]
"""

from __future__ import annotations

import hashlib
import json


def _u01(*parts) -> float:
    """Deterministic uniform(0,1) from arbitrary parts."""
    h = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode())
    return int.from_bytes(h.digest()[:8], "big") / 2**64


class MockBackend:
    """
    Simulates a teacher with a known causal structure.

    If ``marker`` appears anywhere in the rendered prompt, the probability of
    emitting the correct answer is ``p_with``; otherwise ``p_without``.
    Ground-truth direct effect of the marker message is therefore
    ``p_with - p_without``.

    Sampling is deterministic in (prompt, sample_index, seed) but varies across
    sample_index -- which is exactly the property a hash cache must not destroy.
    """

    def __init__(
        self,
        gold: str = "42",
        wrong: str = "17",
        marker: str = "__CAUSAL__",
        p_with: float = 0.90,
        p_without: float = 0.20,
        seed: int = 0,
    ):
        self.gold = gold
        self.wrong = wrong
        self.marker = marker
        self.p_with = p_with
        self.p_without = p_without
        self.seed = seed
        self.calls = 0
        self.generations = 0

    async def generate(
        self,
        messages: list[dict],
        n: int = 1,
        temperature: float = 0.7,
        max_tokens: int | None = None,
        cache_nonce: str | None = None,
    ) -> list[str]:
        self.calls += 1
        self.generations += n
        blob = "\n".join(m.get("content") or "" for m in messages)
        p = self.p_with if self.marker in blob else self.p_without
        out = []
        for i in range(n):
            u = _u01(blob, i, self.seed, temperature, cache_nonce)
            ans = self.gold if u < p else self.wrong
            out.append(f"Reasoning stub (sample {i}).\nThe answer is \\boxed{{{ans}}}.")
        return out

    def ground_truth_delta(self) -> float:
        return self.p_with - self.p_without
