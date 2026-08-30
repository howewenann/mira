"""MIRA Rubric and verifier lifecycle events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from core.api.events.base import EventIdentity


@dataclass(frozen=True, slots=True, kw_only=True)
class RubricEvent(EventIdentity):
    """MIRA Rubric grading and nested verifier lifecycle notification."""

    phase: str
    run_id: str = ""
    pass_number: int = 0
    max_iterations: int = 0
    grader_model: str = ""
    evaluation: Mapping[str, Any] | None = None
    lifecycle: Mapping[str, Any] | None = None
    status: str = ""


__all__ = ["RubricEvent"]
