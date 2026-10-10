"""Commit a forced DeepAgents summary to a completed LangGraph checkpoint."""

from __future__ import annotations

from typing import Any

from deepagents.backends.protocol import FILE_NOT_FOUND

from agent.middleware.compaction import ForcedCompactionPlan, PostTurnCompactionResult


async def commit_forced_compaction(
    agent: Any, thread_id: str, snapshot: Any, plan: ForcedCompactionPlan | None,
) -> PostTurnCompactionResult:
    """Persist a middleware-planned summary without adding a conversational turn."""
    if plan is None:
        return PostTurnCompactionResult(reason="nothing_to_compact")

    config = {"configurable": {"thread_id": thread_id}}
    latest = await agent.aget_state(config)
    if not _same_completed_checkpoint(snapshot, latest):
        raise RuntimeError("Conversation changed while compaction was running")

    summarization = plan.summarization
    backend = summarization._backend
    archive_path = summarization._get_history_path(plan.session_id)
    before = await _archive_content(backend, archive_path)
    try:
        # Native DeepAgents writes the archive section and keeps its format.
        written_path = await summarization._aoffload_to_backend(backend, plan.messages, plan.session_id)
    except Exception:
        await _restore_archive(backend, archive_path, before)
        raise
    if written_path != archive_path:
        await _restore_archive(backend, archive_path, before)
        return PostTurnCompactionResult(reason="archive_failed")

    try:
        written = await _archive_content(backend, archive_path)
    except Exception:
        await _restore_archive(backend, archive_path, before)
        raise
    if written is None or written == before or not written.startswith(before or b""):
        await _restore_archive(backend, archive_path, before)
        raise RuntimeError("Conversation archive was not appended safely")

    latest = await agent.aget_state(config)
    if not _same_completed_checkpoint(snapshot, latest):
        await _restore_archive(backend, archive_path, before, expected=written)
        raise RuntimeError("Conversation changed while compaction was running")

    update = plan.state_update(archive_path)
    try:
        # LangGraph infers the last completed node. Forcing `model` here would
        # schedule MIRA's after-model HITL node on this state-only update.
        await agent.aupdate_state(latest.config or config, update)
    except Exception:
        committed = await agent.aget_state(config)
        if not (
            isinstance(committed.values, dict)
            and committed.values.get("_summarization_event") == update["_summarization_event"]
            and committed.values.get("_summarization_session_id") == plan.session_id
        ):
            await _restore_archive(backend, archive_path, before, expected=written)
            raise
    return PostTurnCompactionResult(
        compacted=True, reason="compacted", file_path=archive_path, summary=plan.summary,
    )


def _same_completed_checkpoint(original: Any, current: Any) -> bool:
    return (
        current.config == original.config
        and current.values == original.values
        and not current.next
        and not current.tasks
    )


async def _archive_content(backend: Any, path: str) -> bytes | None:
    responses = await backend.adownload_files([path])
    if not responses or len(responses) != 1:
        raise RuntimeError("Could not read conversation history before compaction")
    response = responses[0]
    if response.error == FILE_NOT_FOUND:
        return None
    if response.error is not None or response.content is None:
        raise RuntimeError(f"Could not read conversation history: {response.error}")
    return response.content


async def _restore_archive(
    backend: Any, path: str, before: bytes | None, *, expected: bytes | None = None,
) -> None:
    """Undo only the append from this attempt when a checkpoint cannot commit."""
    current = await _archive_content(backend, path)
    if current == before:
        return
    if expected is not None and current != expected:
        raise RuntimeError("Conversation archive changed before rollback")
    if before is not None and (current is None or not current.startswith(before)):
        raise RuntimeError("Conversation archive changed before rollback")
    result = (
        await backend.awrite(path, before.decode("utf-8"))
        if before is not None else await backend.adelete(path)
    )
    if result.error is not None:
        raise RuntimeError(f"Could not restore conversation archive: {result.error}")
