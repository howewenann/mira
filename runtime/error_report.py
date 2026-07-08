"""Automatic error report artifacts for MIRA failures."""

from __future__ import annotations

import json
import platform
import re
import sys
import traceback
from contextlib import suppress
from datetime import datetime
from pathlib import Path
from typing import Any

ERROR_REPORT_ATTR = "__mira_error_report_path__"
_SAFE_SESSION_ID = re.compile(r"[^A-Za-z0-9_.+@=-]+")


def write_error_report(
    exc: BaseException,
    *,
    workspace: Path,
    source: str,
    session_id: str | None = None,
    context: dict[str, Any] | None = None,
) -> Path:
    """Write a copy-pasteable error report and return its timestamped path."""
    existing = getattr(exc, ERROR_REPORT_ATTR, None)
    if existing:
        return Path(str(existing))

    now = datetime.now().astimezone()
    safe_session_id = _safe_session_id(session_id)
    root = workspace.expanduser().resolve() / ".mira" / "_errors"
    session_root = root / safe_session_id
    session_root.mkdir(parents=True, exist_ok=True)

    report_name = f"{now.strftime('%Y%m%d-%H%M%S%z')}-{now.microsecond:06d}.txt"
    report_path = session_root / report_name
    content = _format_report(
        exc,
        timestamp=now.isoformat(),
        session_id=safe_session_id,
        source=source,
        workspace=workspace,
        context=context,
    )
    report_path.write_text(content, encoding="utf-8")
    (root / "latest_error.txt").write_text(content, encoding="utf-8")
    with suppress(AttributeError, TypeError):
        setattr(exc, ERROR_REPORT_ATTR, str(report_path))
    return report_path


def error_report_path(exc: BaseException) -> Path | None:
    """Return the existing report path attached to an exception, if any."""
    existing = getattr(exc, ERROR_REPORT_ATTR, None)
    return Path(str(existing)) if existing else None


def _format_report(
    exc: BaseException,
    *,
    timestamp: str,
    session_id: str,
    source: str,
    workspace: Path,
    context: dict[str, Any] | None,
) -> str:
    report_context = {"workspace": str(workspace), **(context or {})}
    context_text = json.dumps(_json_value(report_context), indent=2, sort_keys=True)
    traceback_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return (
        "MIRA error report\n\n"
        f"Timestamp: {timestamp}\n"
        f"Session ID: {session_id}\n"
        f"Source: {source}\n"
        f"Error: {type(exc).__name__}: {exc}\n"
        f"Python: {sys.version.replace(chr(10), ' ')}\n"
        f"Platform: {platform.platform()}\n\n"
        "Context:\n"
        f"{context_text}\n\n"
        "Traceback:\n"
        f"{traceback_text}"
    )


def _safe_session_id(session_id: str | None) -> str:
    value = str(session_id or "").strip()
    if not value:
        return "unknown-session"
    value = _SAFE_SESSION_ID.sub("_", value)
    value = value.strip("._")
    return value or "unknown-session"


def _json_value(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if isinstance(value, dict):
            return {str(key): _json_value(item) for key, item in value.items()}
        if isinstance(value, list | tuple):
            return [_json_value(item) for item in value]
        return str(value)
