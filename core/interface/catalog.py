"""Detached metadata for frontend command and MCP navigation."""

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class MCPApprovalInfo:
    """Server identity presented before connecting an MCP transport."""

    name: str
    transport: str


@dataclass(frozen=True, slots=True)
class CommandInfo:
    command: str
    usage: str
    description: str = ""
    kind: str = "prompt"
    source: str = "local"
    server: str = ""
    arguments: tuple[Any, ...] = ()
    name: str = ""


@dataclass(frozen=True, slots=True)
class ResourceInfo:
    name: str
    description: str = ""
    uris: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MCPServerSnapshot:
    name: str
    transport: str
    status: str
    usable: bool
    transient: bool
    error: str
    tool_metadata: tuple[Mapping[str, Any], ...]
    prompts: tuple[CommandInfo, ...] | None
    resources: tuple[Any, ...] | None
    skills: tuple[ResourceInfo, ...] | None
    prompt_error: str = ""
    resource_error: str = ""
    skill_error: str = ""
    stderr_tail: str = ""

    @property
    def tools(self) -> tuple[Mapping[str, Any], ...]:
        return self.tool_metadata
