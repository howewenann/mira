"""Ephemeral model guidance for user-referenced local files."""

from __future__ import annotations

from typing import Any

from langchain.agents.middleware.types import AgentMiddleware
from langchain_core.messages import HumanMessage, SystemMessage

from agent.file_references import local_file_references


class FileReferenceMiddleware(AgentMiddleware):
    """Tell the model how to inspect the latest user's visible file mentions."""

    def wrap_model_call(self, request: Any, handler: Any) -> Any:
        return handler(self._request_with_guidance(request))

    async def awrap_model_call(self, request: Any, handler: Any) -> Any:
        return await handler(self._request_with_guidance(request))

    def _request_with_guidance(self, request: Any) -> Any:
        message = _latest_human_message(request.state.get("messages", []))
        references = local_file_references(str(message.text)) if message is not None else []
        if not references:
            return request

        guidance = file_reference_guidance(references)
        current = getattr(request, "system_message", None)
        if current is None:
            system_message = SystemMessage(content=guidance)
        else:
            system_message = current.model_copy(
                update={
                    "content": [
                        *current.content_blocks,
                        {"type": "text", "text": f"\n\n{guidance}"},
                    ]
                }
            )
        return request.override(system_message=system_message)


def file_reference_guidance(references: list[str]) -> str:
    """Build concise read_file guidance for normalized paths."""
    lines = [
        "## User-referenced files",
        "",
        "The user explicitly referenced these local files. Inspect them with read_file",
        "before relying on their contents. If a file is unavailable, handle the normal",
        "read_file error accurately.",
    ]
    for path in references:
        lines.extend(("", f"- {path}", f'  Use read_file(file_path="{path}").'))
    return "\n".join(lines)


def _latest_human_message(messages: list[Any]) -> HumanMessage | None:
    return next((message for message in reversed(messages) if isinstance(message, HumanMessage)), None)


__all__ = ["FileReferenceMiddleware", "file_reference_guidance"]
