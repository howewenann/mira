"""Known provider and filesystem-tool response normalizations."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain.agents.middleware.types import AgentMiddleware, ModelResponse
from langchain_core.messages import AIMessage


class ModelResponseNormalizationMiddleware(AgentMiddleware[Any, Any, Any]):
    """Correct known model-response incompatibilities before MIRA uses them."""

    MODEL_PROVIDER = "anyllm"
    FILE_PATH_TOOLS = {"read_file", "write_file", "edit_file"}

    def __init__(self, workspace: Path) -> None:
        self.workspace = Path(workspace).expanduser().resolve()

    def wrap_model_call(self, request: Any, handler: Any) -> ModelResponse[Any]:
        response = handler(request)
        self._normalize_response(response)
        return response

    async def awrap_model_call(self, request: Any, handler: Any) -> ModelResponse[Any]:
        response = await handler(request)
        self._normalize_response(response)
        return response

    def _normalize_response(self, response: ModelResponse[Any]) -> None:
        for message in response.result:
            if not isinstance(message, AIMessage):
                continue
            self._normalize_metadata(message)
            self._normalize_file_tool_calls(message)

    def _normalize_metadata(self, message: AIMessage) -> None:
        message.response_metadata.setdefault("model_provider", self.MODEL_PROVIDER)

    def _normalize_file_tool_calls(self, message: AIMessage) -> None:
        if not message.tool_calls:
            return
        changed = False
        normalized_calls = []
        for call in message.tool_calls:
            normalized, call_changed = self._normalize_tool_call(call)
            normalized_calls.append(normalized)
            changed = changed or call_changed
        normalized_content, content_changed = self._normalize_content_blocks(message.content)
        changed = changed or content_changed
        if not changed:
            return
        message.tool_calls = normalized_calls
        if content_changed:
            message.content = normalized_content

    def _normalize_tool_call(self, call: Any) -> tuple[Any, bool]:
        if not isinstance(call, dict) or call.get("name") not in self.FILE_PATH_TOOLS:
            return call, False
        args = call.get("args")
        if not isinstance(args, dict):
            return call, False
        normalized_args = dict(args)
        changed = False
        if "file_path" not in normalized_args and "path" in normalized_args:
            normalized_args["file_path"] = normalized_args.pop("path")
            changed = True
        file_path = normalized_args.get("file_path")
        if isinstance(file_path, str):
            normalized_path = self._normalize_workspace_path(file_path)
            if normalized_path != file_path:
                normalized_args["file_path"] = normalized_path
                changed = True
        if not changed:
            return call, False
        return {**call, "args": normalized_args}, True

    def _normalize_content_blocks(self, content: Any) -> tuple[Any, bool]:
        if not isinstance(content, list):
            return content, False
        changed = False
        normalized_blocks = []
        for block in content:
            normalized, block_changed = self._normalize_tool_call(block)
            normalized_blocks.append(normalized)
            changed = changed or block_changed
        return normalized_blocks, changed

    def _normalize_workspace_path(self, value: str) -> str:
        if value.startswith("/"):
            return value
        try:
            path = Path(value).expanduser()
        except (OSError, RuntimeError):
            return value
        if not path.is_absolute():
            return value
        try:
            relative = path.resolve().relative_to(self.workspace)
        except (OSError, RuntimeError, ValueError):
            return value
        return f"/{relative.as_posix()}"


__all__ = ["ModelResponseNormalizationMiddleware"]
