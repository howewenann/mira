"""Copy the effective conversation between independent native graph threads."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import RemoveMessage
from langgraph.graph.message import REMOVE_ALL_MESSAGES
from langgraph.graph import END

from agent.middleware.compaction import normalize_summarization_event


def effective_conversation(agent: Any, values: dict[str, Any]) -> list[Any]:
    """Ask the agent's DeepAgents middleware to project its compacted history."""
    summarization = getattr(agent, "mira_summarization", None)
    if summarization is None:
        return list(values.get("messages") or [])
    return summarization._apply_event_to_messages(
        list(values.get("messages") or []),
        normalize_summarization_event(values.get("_summarization_event")),
    )


def _conversation_key(messages: list[Any]) -> list[dict[str, Any]]:
    # DeepAgents recreates the summary message on projection, so its generated
    # message ID is not stable. All provider-visible and tool linkage fields are.
    return [message.model_dump(exclude={"id"}) for message in messages]


def _completed(state: Any, thread_id: str) -> None:
    if state.next or state.tasks:
        raise RuntimeError(f"Cannot synchronize {thread_id}: checkpoint has pending execution or interrupts")


async def synchronize_conversation(
    source_agent: Any, source_id: str, target_agent: Any, target_id: str,
) -> bool:
    """Replace messages only, leaving destination todos and other channels intact."""
    source_config = {"configurable": {"thread_id": source_id}}
    target_config = {"configurable": {"thread_id": target_id}}
    source = await source_agent.aget_state(source_config)
    target = await target_agent.aget_state(target_config)
    _completed(source, source_id)
    _completed(target, target_id)
    inherited = effective_conversation(source_agent, source.values)
    existing = effective_conversation(target_agent, target.values)
    if _conversation_key(existing) == _conversation_key(inherited):
        if (await source_agent.aget_state(source_config)).config != source.config or (
            await target_agent.aget_state(target_config)
        ).config != target.config:
            raise RuntimeError("Conversation changed during handoff")
        return False
    event = source.values.get("_summarization_event")
    archive_path = event.get("file_path") if isinstance(event, dict) else None
    backend = getattr(target_agent, "mira_backend", None)
    if archive_path and backend is not None:
        archive = await backend.adownload_files([archive_path])
        if len(archive) != 1 or archive[0].error is not None or not archive[0].content:
            raise RuntimeError("The compacted conversation archive is unavailable to the destination")
    latest_source = await source_agent.aget_state(source_config)
    latest_target = await target_agent.aget_state(target_config)
    if latest_source.config != source.config or latest_source.values != source.values:
        raise RuntimeError("Conversation source changed during handoff")
    if latest_target.config != target.config or latest_target.values != target.values:
        raise RuntimeError("Conversation destination changed during handoff")
    _completed(latest_source, source_id)
    _completed(latest_target, target_id)
    await target_agent.aupdate_state(
        target.config or target_config,
        {
            "messages": [RemoveMessage(id=REMOVE_ALL_MESSAGES), *inherited],
            "_summarization_event": None,
            "_summarization_session_id": None,
        },
    )
    committed = await target_agent.aget_state(target_config)
    if committed.next or committed.tasks:
        # A state update on a fresh thread is attributed to START and schedules
        # its first model node. This update is a completed handoff, not a turn.
        await target_agent.aupdate_state(committed.config or target_config, None, as_node=END)
        committed = await target_agent.aget_state(target_config)
    _completed(committed, target_id)
    if _conversation_key(effective_conversation(target_agent, committed.values)) != _conversation_key(inherited):
        raise RuntimeError("Conversation handoff did not commit the source's effective history")
    if any(
        committed.values.get(key) != value for key, value in target.values.items()
        if key not in {"messages", "_summarization_event", "_summarization_session_id"}
    ):
        raise RuntimeError("Conversation handoff changed unrelated destination state")
    if (await source_agent.aget_state(source_config)).config != source.config:
        raise RuntimeError("Conversation source changed during handoff; retry the transition")
    return True
