"""Upload local files into MIRA's current project backend."""

from __future__ import annotations

import asyncio
import re
import uuid
from pathlib import Path
from typing import Any

from agent.middleware.file_references import file_reference_token


_UNSAFE_SCOPE_CHARACTERS = re.compile(r"[^A-Za-z0-9._+-]+")


async def ingest_local_files(
    paths: list[Path],
    project_backend: Any,
    thread_id: str,
) -> tuple[list[str], list[str]]:
    """Upload local files and return successful ``@`` tokens and failures."""
    references: list[str] = []
    failures: list[str] = []
    scope = safe_upload_scope(thread_id)

    for source in paths:
        source = Path(source)
        destination = f"/.mira/_uploads/{scope}/{uuid.uuid4().hex}/{source.name}"
        try:
            content = await asyncio.to_thread(source.read_bytes)
        except Exception as exc:
            failures.append(f"{source.name}: {concise_error(exc)}")
            continue

        try:
            responses = await project_backend.aupload_files([(destination, content)])
        except Exception as exc:
            failures.append(f"{source.name}: {concise_error(exc)}")
            continue

        response = responses[0] if responses else None
        if response is None:
            failures.append(f"{source.name}: upload returned no result")
            continue
        error = getattr(response, "error", None)
        if error is not None:
            failures.append(f"{source.name}: {error}")
            continue

        uploaded_path = str(getattr(response, "path", "") or destination)
        references.append(file_reference_token(uploaded_path))

    return references, failures


async def clear_local_uploads(
    project_backend: Any,
    thread_id: str | None = None,
) -> bool:
    """Delete current-session or all upload state through the project backend."""
    target = "/.mira/_uploads"
    if thread_id is not None:
        target = f"{target}/{safe_upload_scope(thread_id)}"
    result = await project_backend.adelete(target)
    error = getattr(result, "error", None)
    if error is None:
        return True
    normalized = str(error).casefold()
    if any(marker in normalized for marker in ("not found", "file_not_found", "path_not_found")):
        return False
    raise RuntimeError(str(error))


def safe_upload_scope(thread_id: str) -> str:
    """Return one filesystem-safe virtual path segment for an upload scope."""
    value = _UNSAFE_SCOPE_CHARACTERS.sub("_", str(thread_id or "")).strip(".")
    return value or "session"


def concise_error(error: BaseException) -> str:
    """Return a compact, useful local-file failure description."""
    if isinstance(error, OSError) and error.strerror:
        return error.strerror
    detail = str(error).strip()
    return detail or type(error).__name__


__all__ = ["clear_local_uploads", "ingest_local_files", "safe_upload_scope"]
