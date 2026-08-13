"""
Disk cache for counterfactual outcomes.

Provides an append-only JSONL cache with SHA256-based keying
for storing and retrieving counterfactual outcomes.
"""

import hashlib
import json
from pathlib import Path
from typing import Any


class OutcomeCache:
    """
    Append-only JSONL cache for counterfactual outcomes.

    Uses SHA256 of the request payload as the key for deduplication
    and efficient lookup.
    """

    def __init__(self, cache_dir: Path):
        """
        Initialize the cache.

        Args:
            cache_dir: Directory to store cache files.
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._index: dict[str, dict[str, Any]] = {}
        self._load_index()

    def _load_index(self) -> None:
        """Load the cache index from existing cache files."""
        index_file = self.cache_dir / "index.json"
        if index_file.exists():
            with open(index_file) as f:
                self._index = json.load(f)

    def _save_index(self) -> None:
        """Save the cache index to disk."""
        index_file = self.cache_dir / "index.json"
        with open(index_file, "w") as f:
            json.dump(self._index, f, indent=2)

    def _make_key(self, payload: dict[str, Any]) -> str:
        """
        Create a cache key from the payload.

        Args:
            payload: Request payload dictionary.

        Returns:
            SHA256 hex digest of the payload.
        """
        payload_str = json.dumps(payload, sort_keys=True)
        return hashlib.sha256(payload_str.encode()).hexdigest()

    def get(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        """
        Get cached outcome for a payload.

        Args:
            payload: Request payload dictionary.

        Returns:
            Cached outcome or None if not found.
        """
        key = self._make_key(payload)
        if key in self._index:
            cache_file = self.cache_dir / self._index[key]["filename"]
            if cache_file.exists():
                with open(cache_file) as f:
                    return json.load(f)
        return None

    def put(self, payload: dict[str, Any], outcome: dict[str, Any]) -> str:
        """
        Store an outcome in the cache.

        Args:
            payload: Request payload dictionary.
            outcome: Outcome dictionary to cache.

        Returns:
            The cache key for this entry.
        """
        key = self._make_key(payload)

        # Create cache file
        filename = f"{key}.json"
        cache_file = self.cache_dir / filename

        with open(cache_file, "w") as f:
            json.dump(outcome, f, indent=2)

        # Update index
        self._index[key] = {
            "filename": filename,
            "payload_key": key,
        }
        self._save_index()

        return key

    def append_outcome(self, payload: dict[str, Any], outcome: dict[str, Any]) -> None:
        """
        Append an outcome to the JSONL log.

        This is an alternative append-only storage format.

        Args:
            payload: Request payload dictionary.
            outcome: Outcome dictionary to store.
        """
        log_file = self.cache_dir / "outcomes.jsonl"
        entry = {
            "key": self._make_key(payload),
            "payload": payload,
            "outcome": outcome,
        }
        with open(log_file, "a") as f:
            f.write(json.dumps(entry) + "\n")

    def has(self, payload: dict[str, Any]) -> bool:
        """
        Check if a payload is in the cache.

        Args:
            payload: Request payload dictionary.

        Returns:
            True if the payload is cached.
        """
        return self._make_key(payload) in self._index

    def clear(self) -> None:
        """Clear all cached entries."""
        for entry in self._index.values():
            cache_file = self.cache_dir / entry["filename"]
            if cache_file.exists():
                cache_file.unlink()
        self._index = {}
        self._save_index()