"""Test-only bridge from legacy renderer doubles to the Core consumer API."""

from __future__ import annotations

from typing import Any

from core.execution.runner import TurnResult, run_turn
from core.execution.turns import run_user_turn as run_core_user_turn
from ui.shared.adapter import RendererAdapter


async def run_user_turn(
    *,
    renderer: Any | None = None,
    frontend: Any | None = None,
    **kwargs: Any,
) -> TurnResult:
    """Run a Core interaction using existing renderer-shaped test doubles."""
    return await run_core_user_turn(
        frontend=frontend if frontend is not None else RendererAdapter(renderer),
        turn_runner=run_turn,
        **kwargs,
    )


__all__ = ["run_user_turn"]
