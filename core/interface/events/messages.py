"""Message events preserving LangChain-normalized content blocks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from core.interface.events.base import EventIdentity


@dataclass(frozen=True, slots=True, kw_only=True)
class MessageEvent(EventIdentity):
    """One ordered user, assistant, or visible-reasoning message update."""

    phase: Literal["user", "content", "reasoning", "end", "discard_reasoning"]
    text: str = ""
    content_blocks: tuple[Any, ...] = ()
    mode: str = ""
    attachments: tuple[Mapping[str, str], ...] = ()


__all__ = ["MessageEvent"]
