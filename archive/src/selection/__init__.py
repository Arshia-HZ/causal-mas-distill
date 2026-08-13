"""Message selection strategies (one per experimental arm)."""

from .base import Selector

# Back-compat alias: several modules referenced the old name.
BaseSelector = Selector

__all__ = ["Selector", "BaseSelector"]
