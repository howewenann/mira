"""JSON-backed session metadata storage."""

from __future__ import annotations

import contextlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from session.context import normalize_session
from session.dashboard import normalize_dashboard
from session.subagent_runs import reconcile_stale_runs


def new_session_id() -> str:
    """Return a sortable, readable default session id."""
    timestamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S%z")
    suffix = uuid.uuid4().hex[:8]
    return f"{timestamp}-{suffix}"


class SessionStore:
    """Persist durable session records as JSON files."""

    def __init__(self, root: Path) -> None:
        """Create the session directory if it does not exist yet."""
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def load(
        self,
        session_id: str | None,
        resume: bool,
        workspace: Path,
    ) -> dict[str, Any]:
        """Load an explicit session, resume the latest, or create a new one."""
        if session_id:
            path = self.path(session_id)
            if path.exists():
                return self.read(path)

            record = self.new(session_id=session_id, workspace=workspace)
            self.save(record)
            return record

        if resume:
            latest = self.latest()
            if latest:
                return self.read(latest)

        record = self.new(session_id=None, workspace=workspace)
        self.save(record)
        return record

    def new(
        self,
        session_id: str | None,
        workspace: Path,
    ) -> dict[str, Any]:
        """Build a new session record without writing it to disk."""
        now = datetime.now(timezone.utc).isoformat()

        return {
            "id": session_id or new_session_id(),
            "title": "Untitled session",
            "custom_title": "",
            "pinned": False,
            "workspace": str(workspace),
            "created_at": now,
            "updated_at": now,
            "turns": 0,
            "dashboard": normalize_dashboard(None),
            "current_plan": None,
            "current_goal": None,
            "events": [],
            "runs": [],
        }

    def save(self, record: dict[str, Any]) -> None:
        """Update the timestamp and write the session JSON file."""
        self._write(record, update_timestamp=True)

    def save_metadata(self, record: dict[str, Any]) -> None:
        """Write pin/title metadata without changing conversation recency."""
        self._write(record, update_timestamp=False)

    def _write(self, record: dict[str, Any], *, update_timestamp: bool) -> None:
        """Normalize and persist one record with explicit timestamp ownership."""
        resume_context_pending = record.get("resume_context_pending") is True
        if update_timestamp:
            record["updated_at"] = datetime.now(timezone.utc).isoformat()
        normalized = normalize_session(record)
        self.path(str(record["id"])).write_text(json.dumps(normalized, indent=2), encoding="utf-8")
        record.clear()
        record.update(normalized)
        if resume_context_pending:
            record["resume_context_pending"] = True

    def read(self, path: Path) -> dict[str, Any]:
        """Read a session record from a JSON file."""
        record = json.loads(path.read_text(encoding="utf-8"))
        record = normalize_session(record)
        if reconcile_stale_runs(record):
            self._write(record, update_timestamp=False)
        return record

    def latest(self) -> Path | None:
        """Return the most recently modified session file, if any exist."""
        paths = list(self.root.glob("*.json"))
        if not paths:
            return None

        return max(paths, key=self._updated_at_timestamp)

    def _updated_at_timestamp(self, path: Path) -> float:
        """Read conversation recency without treating metadata writes as turns."""
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
            text = str(record.get("updated_at") or record.get("created_at") or "")
            if text.endswith("Z"):
                text = text[:-1] + "+00:00"
            return datetime.fromisoformat(text).timestamp()
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return path.stat().st_mtime

    def delete(self, session_id: str) -> bool:
        """Delete one saved session JSON file."""
        path = self.path(session_id)
        if not path.exists():
            return False
        path.unlink()
        return True

    def clear_all(self) -> int:
        """Delete all saved session JSON files under the session root."""
        count = 0
        for path in list(self.root.glob("*.json")):
            path.unlink()
            count += 1
        return count

    def clear_compactions(self) -> int:
        """Delete saved DeepAgents conversation-history files for this workspace."""
        root = self.root.parent / "conversation_history"
        if not root.exists() or not root.is_dir():
            return 0

        count = 0
        for path in sorted(root.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
                count += 1
            elif path.is_dir():
                with contextlib.suppress(OSError):
                    path.rmdir()
        return count

    def path(self, session_id: str) -> Path:
        """Return the JSON path for a session id."""
        return self.root / f"{session_id}.json"
