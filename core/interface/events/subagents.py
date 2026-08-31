"""Subagent lifecycle events with graph provenance."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from core.interface.events.base import EventIdentity


@dataclass(frozen=True, slots=True, kw_only=True)
class SubagentEvent(EventIdentity):
    """One subagent/task lifecycle update, including graph provenance."""

    phase: Literal[
        "live_start",
        "live_tick",
        "live_stop",
        "delegation_update",
        "start",
        "request_update",
        "finish",
        "cancel",
        "cancel_all",
        "eval_start",
        "eval_finish",
        "eval_cancel",
    ]
    subagent_id: str = ""
    name: str = ""
    task_input: str = ""
    result: str = ""
    origin: str = ""
    eval_id: str = ""
    row_id: str = ""
    model: str = ""
    label: str = ""


__all__ = ["SubagentEvent"]
