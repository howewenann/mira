"""Small normalization helpers shared by durable session artifacts."""

from __future__ import annotations

from typing import Any

LIFECYCLE_FIELDS = {"rubric_enabled", "rubric_iterations", "attempts"}
OPTIONAL_ARTIFACT_TEXT = {"last_rubric_status", "completion_source"}


def valid_artifact(
    value: Any,
    *,
    fields: frozenset[str],
    list_fields: tuple[str, ...],
    statuses: set[str],
) -> bool:
    """Validate the shared exact shape of a current Plan or Goal artifact."""
    if not isinstance(value, dict) or set(value) != fields:
        return False
    text_fields = fields - set(list_fields) - LIFECYCLE_FIELDS
    if any(not isinstance(value[field], str) for field in text_fields):
        return False
    if any(not value[field].strip() for field in text_fields - OPTIONAL_ARTIFACT_TEXT):
        return False
    if any(
        not isinstance(value[field], list)
        or not value[field]
        or any(not isinstance(item, str) or not item.strip() for item in value[field])
        for field in list_fields
    ):
        return False
    iterations = value["rubric_iterations"]
    attempts = value["attempts"]
    return (
        value["status"] in statuses
        and isinstance(value["rubric_enabled"], bool)
        and isinstance(iterations, int)
        and not isinstance(iterations, bool)
        and 1 <= iterations <= 20
        and isinstance(attempts, int)
        and not isinstance(attempts, bool)
        and attempts >= 0
    )
