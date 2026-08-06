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
        messages = request.state.get("messages", [])
        message = _latest_human_message(messages)
        references = local_file_references(str(message.text)) if message is not None else []
        attachments = _mcp_attachments(messages)
        if not references and not attachments:
            return request

        guidance_parts = []
        if references:
            guidance_parts.append(file_reference_guidance(references))
        if attachments:
            guidance_parts.append(mcp_resource_guidance(attachments))
        guidance = "\n\n".join(guidance_parts)
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


def mcp_resource_guidance(attachments: list[dict[str, str]]) -> str:
    lines = [
        "## User-attached MCP resources",
        "",
        "The user explicitly attached these resources. Their contents were not injected.",
        "Use read_mcp_resource with the exact server and URI when their contents are needed.",
    ]
    for item in attachments:
        lines.extend(
            (
                "",
                f"- {item.get('token', '')}",
                f'  Use read_mcp_resource(server="{item.get("server", "")}", uri="{item.get("uri", "")}").',
            )
        )
    return "\n".join(lines)


def _mcp_attachments(messages: list[Any]) -> list[dict[str, str]]:
    found: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for message in messages:
        metadata = getattr(message, "additional_kwargs", None)
        values = metadata.get("mira_mcp_attachments", []) if isinstance(metadata, dict) else []
        for item in values:
            if not isinstance(item, dict) or item.get("kind") != "mcp_resource":
                continue
            key = (str(item.get("server") or ""), str(item.get("uri") or ""))
            if key in seen:
                continue
            seen.add(key)
            found.append({str(k): str(v) for k, v in item.items()})
    return found


__all__ = ["FileReferenceMiddleware", "file_reference_guidance", "mcp_resource_guidance"]
