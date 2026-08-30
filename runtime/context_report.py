"""Estimated composition of MIRA's current main-model context."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence


@dataclass(frozen=True, slots=True)
class ContextReportObservation:
    """The effective MIRA request immediately before DeepAgents' tail middleware."""

    system_prompt: str
    messages: tuple[Any, ...]
    tools: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class ContextReportRow:
    """One estimated contributor or nested contributor detail."""

    label: str
    tokens: int | None
    detail: str = ""
    children: tuple[ContextReportRow, ...] = ()


@dataclass(frozen=True, slots=True)
class ContextReportMCPServer:
    """One snapshotted MCP server and its mode-visible tools."""

    name: str
    status: str
    error: str = ""
    tools: tuple[Any, ...] = ()


@dataclass(frozen=True, slots=True)
class ContextReport:
    """A structured estimate alongside MIRA's authoritative live measurement."""

    rows: tuple[ContextReportRow, ...]
    injected_tokens: int
    conversation_tokens: int
    accounted_tokens: int
    current_tokens: int | None
    limit_tokens: int | None
    estimation_delta: int | None

    @property
    def estimated_tokens(self) -> int:
        """Return the mutually exclusive contributor total used for shares."""
        return sum(row.tokens or 0 for row in self.rows)

    @property
    def share_total(self) -> int | None:
        """Return a complete total, or None when any contributor is unavailable."""
        if any(row.tokens is None for row in self.rows):
            return None
        return self.estimated_tokens


def estimate_text_tokens(text: str) -> int:
    """Estimate tokens at approximately four characters per token."""
    return math.ceil(len(text) / 4)


def observe_context_inputs(
    messages: Sequence[Any] | None,
    system_message: Any,
    tools: Sequence[Any] | None,
) -> ContextReportObservation:
    """Capture immutable request containers at MIRA's live request boundary."""
    return ContextReportObservation(
        system_prompt=message_text(system_message),
        messages=tuple(messages or ()),
        tools=tuple(tools or ()),
    )


def format_memory_prompt(contents: Mapping[str, str], sources: Sequence[str]) -> str:
    """Reconstruct the exact DeepAgents AGENTS.md memory fragment."""
    from deepagents.middleware.memory import MemoryMiddleware

    middleware = MemoryMiddleware(backend=object(), sources=list(sources))  # type: ignore[arg-type]
    return middleware._format_agent_memory(dict(contents))  # noqa: SLF001


def format_skills_prompt(
    skills: Sequence[Mapping[str, Any]],
    sources: Sequence[str],
    errors: Sequence[str] = (),
) -> str:
    """Reconstruct the exact DeepAgents progressive-disclosure Skills fragment."""
    from deepagents.middleware.skills import SkillsMiddleware

    middleware = SkillsMiddleware(backend=object(), sources=list(sources))  # type: ignore[arg-type]
    return middleware.system_prompt_template.format(
        skills_locations=middleware._format_skills_locations(),  # noqa: SLF001
        skills_load_warnings=middleware._format_skills_load_warnings(list(errors)),  # noqa: SLF001
        skills_list=middleware._format_skills_list(list(skills)),  # noqa: SLF001
    )


def build_context_report(
    *,
    observation: ContextReportObservation | None,
    current_tokens: int | None,
    limit_tokens: int | None,
    base_system_prompt: str = "",
    memory_prompt: str | None = None,
    memory_files: int = 0,
    skills_prompt: str | None = None,
    skills_count: int = 0,
    tool_metadata: Sequence[Mapping[str, str]] = (),
    mcp_servers: Sequence[ContextReportMCPServer] = (),
    fallback_messages: Sequence[Any] = (),
) -> ContextReport:
    """Build five mutually exclusive contributor estimates from one snapshot."""
    observed_system = observation.system_prompt if observation is not None else base_system_prompt
    instructions, observed_memory, observed_skills = split_system_prompt(
        observed_system,
        memory_prompt=memory_prompt,
        skills_prompt=skills_prompt,
    )
    effective_memory = memory_prompt if memory_prompt is not None else observed_memory
    effective_skills = skills_prompt if skills_prompt is not None else observed_skills

    tools = observation.tools if observation is not None else tuple(
        tool for server in mcp_servers for tool in server.tools
    )
    tool_rows = tool_schema_rows(
        tools,
        tool_metadata,
        mcp_servers,
        complete=observation is not None,
    )
    tools_tokens = sum(row.tokens or 0 for row in tool_rows) if observation is not None else None

    messages = observation.messages if observation is not None else tuple(fallback_messages)
    conversation_tokens = estimate_conversation_tokens(messages)
    rows = (
        ContextReportRow(
            "Instructions",
            estimate_text_tokens(instructions),
            "Effective model instructions, excluding Memory and Skills.",
        ),
        ContextReportRow(
            "Memory",
            None if effective_memory is None else estimate_text_tokens(effective_memory),
            _count_detail(memory_files, "memory file"),
        ),
        ContextReportRow(
            "Skills",
            None if effective_skills is None else estimate_text_tokens(effective_skills),
            _count_detail(skills_count, "loaded skill"),
        ),
        ContextReportRow(
            "Tools",
            tools_tokens,
            "Provider-facing schemas from the model-visible tool surface.",
            tool_rows,
        ),
        ContextReportRow(
            "Conversation",
            conversation_tokens,
            _count_detail(len(messages), "message"),
        ),
    )

    injected_tokens = sum(row.tokens or 0 for row in rows[:-1])
    accounted_tokens = sum(row.tokens or 0 for row in rows)
    trusted_current = positive_int_or_none(current_tokens)
    trusted_limit = positive_int_or_none(limit_tokens)
    return ContextReport(
        rows=rows,
        injected_tokens=injected_tokens,
        conversation_tokens=conversation_tokens,
        accounted_tokens=accounted_tokens,
        current_tokens=trusted_current,
        limit_tokens=trusted_limit,
        estimation_delta=(trusted_current - accounted_tokens if trusted_current is not None else None),
    )


async def build_agent_context_report(
    *,
    agent: Any,
    thread_id: str,
    observation: ContextReportObservation | None,
    current_tokens: int | None,
    limit_tokens: int | None,
    tool_metadata: Sequence[Mapping[str, str]],
    mcp_servers: Sequence[ContextReportMCPServer],
) -> ContextReport:
    """Snapshot one compiled agent and build its local context audit."""
    raw_config = getattr(agent, "mira_context_report_config", {})
    config = dict(raw_config) if isinstance(raw_config, Mapping) else {}
    memory_sources = tuple(config.get("memory_sources") or ())
    skill_sources = tuple(config.get("skill_sources") or ())

    state: dict[str, Any] = {}
    getter = getattr(agent, "aget_state", None)
    if callable(getter) and thread_id:
        try:
            snapshot = await getter({"configurable": {"thread_id": thread_id}})
        except Exception:
            snapshot = None
        values = getattr(snapshot, "values", None)
        if isinstance(values, dict):
            state = values

    memory_contents = state.get("memory_contents")
    memory_prompt: str | None = ""
    if memory_sources:
        memory_prompt = (
            format_memory_prompt(memory_contents, memory_sources)
            if isinstance(memory_contents, Mapping)
            else None
        )
    skills = state.get("skills_metadata")
    skill_errors = state.get("skills_load_errors")
    skills_prompt: str | None = ""
    if skill_sources:
        skills_prompt = (
            format_skills_prompt(
                skills,
                skill_sources,
                skill_errors if isinstance(skill_errors, list) else (),
            )
            if isinstance(skills, list)
            else None
        )
    fallback_messages = state.get("messages")
    return build_context_report(
        observation=observation,
        current_tokens=current_tokens,
        limit_tokens=limit_tokens,
        base_system_prompt=str(config.get("system_prompt") or ""),
        memory_prompt=memory_prompt,
        memory_files=len(memory_sources),
        skills_prompt=skills_prompt,
        skills_count=len(skills) if isinstance(skills, list) else 0,
        tool_metadata=tool_metadata,
        mcp_servers=mcp_servers,
        fallback_messages=fallback_messages if isinstance(fallback_messages, list) else (),
    )


def tool_schema_rows(
    tools: Sequence[Any],
    metadata: Sequence[Mapping[str, str]],
    mcp_servers: Sequence[ContextReportMCPServer],
    *,
    complete: bool = True,
) -> tuple[ContextReportRow, ...]:
    """Categorize exact model-visible schemas, using tool-owned MCP metadata."""
    metadata_by_name = {str(item.get("name") or ""): item for item in metadata}
    built_in: list[Any] = []
    custom: list[Any] = []
    mcp: dict[str, list[Any]] = {}

    for tool in tools:
        name = tool_name(tool)
        info = metadata_by_name.get(name, {})
        owned_mcp = mcp_tool_metadata(tool)
        source = str(info.get("source") or "built-in")
        if owned_mcp is not None:
            mcp.setdefault(str(owned_mcp.get("server") or "unknown"), []).append(tool)
        elif source == "mcp":
            mcp.setdefault(str(info.get("server") or "unknown"), []).append(tool)
        elif source in {"project", "custom"}:
            custom.append(tool)
        else:
            built_in.append(tool)

    built_row = schema_row("Built-in", built_in)
    custom_row = schema_row("Custom", custom)
    if not complete:
        built_row = ContextReportRow("Built-in", None, "available after the first model request")
        custom_row = ContextReportRow("Custom", None, "available after the first model request")
    states = {server.name: server for server in mcp_servers}
    server_rows: list[ContextReportRow] = []
    for name in sorted(set(states) | set(mcp), key=str.casefold):
        server = states.get(name)
        server_tools = mcp.get(name, [])
        status = str(server.status if server is not None else "Available")
        if not server_tools and not _mcp_status_available(status):
            detail = ": ".join(
                part for part in (status.lower(), server.error if server is not None else "") if part
            )
            server_rows.append(ContextReportRow(f"{name} · {status.lower()}", None, detail))
            continue
        row = schema_row(_tool_count_label(name, len(server_tools)), server_tools)
        detail = row.detail
        if server is not None and status == "Partially available" and server.error:
            detail = "; ".join(part for part in (detail, server.error) if part)
        server_rows.append(ContextReportRow(row.label, row.tokens, detail))

    mcp_row = ContextReportRow(
        "MCP",
        sum(row.tokens or 0 for row in server_rows),
        _count_detail(sum(len(values) for values in mcp.values()), "tool"),
        tuple(server_rows),
    )
    return built_row, custom_row, mcp_row


def schema_row(label: str, tools: Sequence[Any]) -> ContextReportRow:
    """Estimate compact provider-facing function schemas for one tool group."""
    schemas: list[str] = []
    failures = 0
    for tool in tools:
        try:
            schema = provider_tool_schema(tool)
            schemas.append(json.dumps(schema, separators=(",", ":"), sort_keys=True, default=str))
        except (TypeError, ValueError):
            failures += 1
    detail = _count_detail(len(tools), "tool")
    if failures:
        detail += f"; {failures} schema unavailable"
    return ContextReportRow(label, estimate_text_tokens("".join(schemas)), detail)


def provider_tool_schema(tool: Any) -> dict[str, Any]:
    """Return LangChain's compact OpenAI-style provider schema."""
    from langchain_core.utils.function_calling import convert_to_openai_tool

    schema = convert_to_openai_tool(tool)
    if not isinstance(schema, dict):
        raise TypeError("tool schema is not a mapping")
    return schema


def estimate_conversation_tokens(messages: Sequence[Any]) -> int:
    """Estimate provider-relevant conversation history fields."""
    text = "".join(
        json.dumps(message_payload(message), separators=(",", ":"), sort_keys=True, default=str)
        for message in messages
    )
    return estimate_text_tokens(text)


def message_payload(message: Any) -> dict[str, Any]:
    """Project one LangChain-like message without cumulative usage metadata."""
    if isinstance(message, Mapping):
        role = message.get("role") or message.get("type") or "message"
        payload: dict[str, Any] = {"role": role, "content": message.get("content", "")}
        for key in ("name", "tool_calls", "tool_call_id"):
            if message.get(key):
                payload[key] = message[key]
        return payload

    payload = {
        "role": getattr(message, "type", message.__class__.__name__),
        "content": getattr(message, "content", ""),
    }
    for key in ("name", "tool_calls", "tool_call_id"):
        value = getattr(message, key, None)
        if value:
            payload[key] = value
    return payload


def split_system_prompt(
    system_prompt: str,
    *,
    memory_prompt: str | None = None,
    skills_prompt: str | None = None,
) -> tuple[str, str | None, str | None]:
    """Separate Memory and Skills fragments once, leaving disjoint instructions."""
    observed_memory = extract_section(system_prompt, "<agent_memory>", "</memory_guidelines>")
    observed_skills = extract_section(
        system_prompt,
        "## Skills System",
        "Remember: Skills make you more capable and consistent. When in doubt, check if a skill exists for the task!",
    )
    residual = system_prompt
    # Prefer exact checkpoint reconstructions so their trailing whitespace is
    # not left behind in Instructions and counted in both categories.
    for fragment in _unique_fragments(memory_prompt, skills_prompt, observed_memory, observed_skills):
        residual = residual.replace(fragment, "", 1)
    return residual, observed_memory, observed_skills


def extract_section(text: str, start_marker: str, end_marker: str) -> str | None:
    """Return one inclusive prompt section when both stable markers exist."""
    start = text.find(start_marker)
    if start < 0:
        return None
    end = text.find(end_marker, start)
    if end < 0:
        return None
    return text[start : end + len(end_marker)]


def contributor_share(tokens: int | None, estimated_total: int) -> float | None:
    """Return a contributor's share of the overall estimated total."""
    if tokens is None or estimated_total <= 0:
        return None
    return (tokens / estimated_total) * 100


def current_context_values(report: ContextReport) -> tuple[str, str]:
    """Return Used / Limit and Usage strings without using cumulative I/O."""
    limit = compact_tokens(report.limit_tokens) if report.limit_tokens is not None else "?"
    if report.current_tokens is None:
        return f"pending / {limit}", "—"
    usage = (
        f"{(report.current_tokens / report.limit_tokens) * 100:.0f}%"
        if report.limit_tokens
        else "—"
    )
    return f"{compact_tokens(report.current_tokens)} / {limit}", usage


def estimated_token_text(tokens: int | None) -> str:
    """Format approximate values consistently for every contributor row."""
    if tokens is None:
        return "—"
    return "0" if tokens == 0 else f"~{compact_tokens(tokens)}"


def share_text(tokens: int | None, estimated_total: int) -> str:
    """Format one overall contributor share."""
    share = contributor_share(tokens, estimated_total)
    return "—" if share is None else f"{share:.0f}%"


def message_text(message: Any) -> str:
    """Flatten a LangChain-like system message to its text content."""
    if message is None:
        return ""
    content = getattr(message, "content", message)
    if isinstance(content, str):
        return content
    if isinstance(content, Sequence) and not isinstance(content, str | bytes | bytearray):
        parts: list[str] = []
        for item in content:
            if isinstance(item, Mapping):
                parts.append(str(item.get("text") or ""))
            else:
                parts.append(str(item))
        return "".join(parts)
    return str(content or "")


def tool_name(tool: Any) -> str:
    """Return a tool name across LangChain and schema-dict shapes."""
    if isinstance(tool, Mapping):
        function = tool.get("function")
        if isinstance(function, Mapping) and function.get("name"):
            return str(function["name"])
        return str(tool.get("name") or "")
    return str(getattr(tool, "name", None) or getattr(tool, "__name__", ""))


def mcp_tool_metadata(tool: Any) -> Mapping[str, Any] | None:
    """Return MCP ownership carried by the converted live LangChain tool."""
    metadata = tool.get("metadata") if isinstance(tool, Mapping) else getattr(tool, "metadata", None)
    if not isinstance(metadata, Mapping):
        return None
    ownership = metadata.get("mira_mcp")
    return ownership if isinstance(ownership, Mapping) else None


def compact_tokens(value: int | None) -> str:
    """Format token values consistently with the status header."""
    if value is None:
        return "?"
    if value >= 1_000_000:
        return f"{value / 1_000_000:.1f}m"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)


def positive_int_or_none(value: Any) -> int | None:
    """Treat zero and invalid values as unavailable observations."""
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _unique_fragments(*fragments: str | None) -> tuple[str, ...]:
    values: list[str] = []
    for fragment in fragments:
        if fragment and fragment not in values:
            values.append(fragment)
    return tuple(values)


def _count_detail(count: int, noun: str) -> str:
    suffix = "" if count == 1 else "s"
    return f"{count} {noun}{suffix}"


def _tool_count_label(name: str, count: int) -> str:
    return f"{name} · {_count_detail(count, 'tool')}"


def _mcp_status_available(status: str) -> bool:
    return status in {"Available", "Partially available"}


__all__ = [
    "ContextReport",
    "ContextReportMCPServer",
    "ContextReportObservation",
    "ContextReportRow",
    "build_agent_context_report",
    "build_context_report",
    "contributor_share",
    "current_context_values",
    "estimate_conversation_tokens",
    "estimate_text_tokens",
    "estimated_token_text",
    "format_memory_prompt",
    "format_skills_prompt",
    "mcp_tool_metadata",
    "observe_context_inputs",
    "share_text",
    "split_system_prompt",
    "tool_schema_rows",
]
