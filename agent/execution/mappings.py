"""Small immutable mapping used by public execution contexts."""

from __future__ import annotations

from typing import Any, TypeVar

Key = TypeVar("Key")
Value = TypeVar("Value")


class ImmutableDict(dict[Key, Value]):
    """A dict-compatible immutable mapping that Pydantic serializes cleanly."""

    def _immutable(self, *_args: Any, **_kwargs: Any) -> None:
        raise TypeError("MIRA execution-context mappings are immutable")

    __delitem__ = _immutable
    __ior__ = _immutable
    __setitem__ = _immutable
    clear = _immutable
    pop = _immutable
    popitem = _immutable
    setdefault = _immutable
    update = _immutable


__all__ = ["ImmutableDict"]
