"""Shared immutable MCP and prompt registry records."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any, Literal

from langchain_core.messages import BaseMessage

MCPStatus = Literal[
    "Available",
    "Partially available",
    "Approval required",
    "Failed",
    "Disabled",
    "Login required",
    "Authenticating",
    "Starting",
    "Restarting",
    "Stopping",
]


@dataclass(frozen=True, slots=True)
class MCPConfigIssue:
    """One whole-file MCP configuration problem shown through Issues."""

    message: str
    display_path: str = ".mira/mcp.json"
    exception_type: str = "MCP configuration"
    missing_module: str = ""
    line_number: int | None = None
    source_line: str = ""


@dataclass(frozen=True, slots=True)
class MCPResource:
    token: str
    server: str
    uri: str
    name: str = ""
    description: str = ""
    mime_type: str = ""

    def attachment(self) -> dict[str, str]:
        return {
            "kind": "mcp_resource",
            "token": self.token,
            "server": self.server,
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mime_type": self.mime_type,
        }


@dataclass(frozen=True, slots=True)
class PromptArgument:
    name: str
    required: bool = True


PromptResolver = Callable[[list[str]], Awaitable[list[BaseMessage]]]


@dataclass(frozen=True, slots=True)
class PromptSpec:
    command: str
    description: str
    arguments: tuple[PromptArgument, ...]
    source: Literal["local", "mcp"]
    resolver: PromptResolver
    server: str = ""

    @property
    def usage(self) -> str:
        placeholders = " ".join(f"<{item.name}>" for item in self.arguments)
        return f"{self.command} {placeholders}".rstrip()


@dataclass(frozen=True, slots=True)
class PreparedPrompt:
    display_text: str
    messages: list[BaseMessage]


@dataclass(slots=True)
class MCPServerState:
    name: str
    transport: Literal["stdio", "http"]
    config: dict[str, Any]
    fingerprint: str
    status: MCPStatus = "Disabled"
    error: str = ""
    tools: list[Any] = field(default_factory=list)
    tool_metadata: list[dict[str, str]] = field(default_factory=list)
    prompts: list[PromptSpec] | None = None
    resources: list[MCPResource] | None = None
    prompt_error: str = ""
    resource_error: str = ""
    session: Any = None
    exit_stack: Any = None

    @property
    def usable(self) -> bool:
        return self.status in {"Available", "Partially available"}

    @property
    def transient(self) -> bool:
        return self.status in {"Authenticating", "Starting", "Restarting", "Stopping"}
