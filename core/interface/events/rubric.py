"""MIRA Rubric and verifier lifecycle events."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

from core.interface.events.base import EventIdentity

RubricPhase = Literal["verifying", "grading", "lifecycle", "finish", "cancel", "status"]


@dataclass(frozen=True, slots=True, kw_only=True)
class RubricEvent(EventIdentity):
    """MIRA Rubric grading and nested verifier lifecycle notification.

    ``verifying`` and ``grading`` open visible phases; ``lifecycle`` carries a
    native nested verifier event; ``finish`` carries the evaluation; ``status``
    reconciles its final result; and ``cancel`` ends unfinished activity.
    """

    phase: RubricPhase
    run_id: str = ""
    pass_number: int = 0
    max_iterations: int = 0
    grader_model: str = ""
    evaluation: Mapping[str, Any] | None = None
    lifecycle: Mapping[str, Any] | None = None
    status: str = ""


__all__ = ["RubricEvent", "RubricPhase"]
