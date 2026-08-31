"""Stable identities for framework-internal nested agent graphs."""

from __future__ import annotations

RUBRIC_VERIFIER_GRAPH = "rubric_verifier"
RUBRIC_GRADER_GRAPH = "rubric_grader"

INTERNAL_RUBRIC_GRAPHS = frozenset(
    {
        RUBRIC_VERIFIER_GRAPH,
        RUBRIC_GRADER_GRAPH,
    }
)
