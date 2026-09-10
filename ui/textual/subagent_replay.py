"""Retrospective subagent projections reconstructed from saved session JSON."""

from __future__ import annotations

from typing import Any

from session.context import normalize_session
from session.subagent_runs import get_run, runs_for_anchor


def persisted_session(store: Any, current: dict[str, Any]) -> dict[str, Any]:
    """Read the active session file, with a narrow fallback for test frontends."""
    session_id = str(current.get("id") or "")
    path_method = getattr(store, "path", None)
    read_method = getattr(store, "read", None)
    if session_id and callable(path_method) and callable(read_method):
        path = path_method(session_id)
        if path.exists():
            return read_method(path)
    return normalize_session(current)


def restore_anchor(panel: Any, session: dict[str, Any], anchor_id: str) -> list[dict[str, Any]]:
    """Restore the shared panel from only the runs owned by one anchor."""
    runs = runs_for_anchor(session, anchor_id)
    panel.restore(runs)
    return runs


def replay_run(session: dict[str, Any], run_id: str) -> dict[str, Any] | None:
    """Return one inspector input from a freshly loaded session record."""
    return get_run(session, run_id)
