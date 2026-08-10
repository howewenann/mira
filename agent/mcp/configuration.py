"""Strict project MCP configuration loading and safe approval previews."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.mcp.models import MCPServerState
from agent.resources.paths import MCP_DIR, PROJECT_DIR
from config.interpolation import resolve_environment
from runtime.issues import Issue

MCP_FILE = "mcp.json"


@dataclass(frozen=True, slots=True)
class MCPConfiguration:
    exists: bool
    valid: bool
    servers: dict[str, MCPServerState]
    issues: tuple[Issue, ...] = ()

    @property
    def issue(self) -> Issue | None:
        return self.issues[0] if self.issues else None


def mcp_path(workspace: Path) -> Path:
    return workspace.expanduser().resolve() / PROJECT_DIR / MCP_DIR / MCP_FILE


def load_mcp_configuration(
    workspace: Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> MCPConfiguration:
    """Load the common ``mcpServers`` shape without making startup fatal."""
    path = mcp_path(workspace)
    if not path.exists():
        return MCPConfiguration(False, True, {})
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        return MCPConfiguration(True, False, {}, (_configuration_issue(_error_text(error)),))
    if not isinstance(raw, dict) or not isinstance(raw.get("mcpServers"), dict):
        return MCPConfiguration(
            True,
            False,
            {},
            (_configuration_issue("top level must be an object containing an mcpServers object"),),
        )

    servers: dict[str, MCPServerState] = {}
    issues: list[Issue] = []
    for alias, definition in raw["mcpServers"].items():
        name = str(alias) if isinstance(alias, str) else ""
        if not name.strip():
            continue
        try:
            transport, effective = normalize_server_definition(definition)
            runtime_config = resolve_environment(effective, environ=environ)
            _validate_runtime_config(transport, runtime_config)
            state = MCPServerState(
                name=name,
                transport=transport,
                config=effective,
                fingerprint=configuration_fingerprint(effective),
                runtime_config=runtime_config,
            )
        except ValueError as error:
            guessed = "http" if isinstance(definition, dict) and definition.get("type") == "http" else "stdio"
            effective = dict(definition) if isinstance(definition, dict) else {}
            state = MCPServerState(
                name=name,
                transport=guessed,
                config=effective,
                fingerprint=configuration_fingerprint(effective),
                status="Failed",
                error=str(error),
            )
            issues.append(
                Issue(
                    "MCP",
                    f"Invalid MCP server configuration: {name}",
                    f".mira/mcp/mcp.json: mcpServers.{name}",
                    str(error),
                    "Correct the server configuration and run /reload.",
                )
            )
        servers[name] = state
    return MCPConfiguration(True, True, servers, tuple(issues))


def normalize_server_definition(raw: Any) -> tuple[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        raise ValueError("server definition must be an object")
    kind = raw.get("type")
    if kind == "http":
        url = raw.get("url")
        headers = raw.get("headers", {})
        if not isinstance(url, str) or not url.strip():
            raise ValueError("HTTP server requires a non-empty url")
        if not isinstance(headers, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in headers.items()):
            raise ValueError("HTTP headers must contain string names and values")
        return "http", {"transport": "http", "url": url, "headers": dict(headers)}
    if kind not in {None, "stdio"}:
        raise ValueError("server type must be http or omitted for stdio")
    command = raw.get("command")
    args = raw.get("args", [])
    env = raw.get("env", {})
    if not isinstance(command, str) or not command.strip():
        raise ValueError("stdio server requires a non-empty command")
    if not isinstance(args, list) or not all(isinstance(item, str) for item in args):
        raise ValueError("stdio args must be a list of strings")
    if not isinstance(env, dict) or not all(isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        raise ValueError("stdio env must contain string names and values")
    return "stdio", {"transport": "stdio", "command": command, "args": list(args), "env": dict(env)}


def _validate_runtime_config(transport: str, config: dict[str, Any]) -> None:
    if transport == "http" and not str(config.get("url", "")).strip():
        raise ValueError("HTTP server url resolved to an empty value")
    if transport == "stdio" and not str(config.get("command", "")).strip():
        raise ValueError("stdio server command resolved to an empty value")


def configuration_fingerprint(effective: dict[str, Any]) -> str:
    """Hash the exact effective config; raw secret values are never persisted."""
    payload = json.dumps(effective, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def approval_preview(state: MCPServerState) -> str:
    config = state.config
    if state.transport == "stdio":
        args = " ".join(str(item) for item in config.get("args", []))
        target = f"{config.get('command', '')} {args}".strip()
        names = ", ".join(sorted(config.get("env", {}))) or "none"
        return (
            "MIRA will launch this MCP server and keep its session active.\n"
            f"Command: {target}\nEnvironment variable names: {names}"
        )
    names = ", ".join(sorted(config.get("headers", {}))) or "none"
    return (
        "MIRA will use this remote MCP endpoint.\n"
        f"URL: {config.get('url', '')}\nHeader names: {names}"
    )


def adapter_connection(state: MCPServerState) -> dict[str, Any]:
    config = dict(state.connection_config)
    if config.get("transport") == "http":
        config["transport"] = "streamable_http"
    return config


def _error_text(error: BaseException) -> str:
    message = str(error).strip()
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def _configuration_issue(message: str) -> Issue:
    return Issue(
        "MCP",
        "Invalid MCP configuration",
        ".mira/mcp/mcp.json",
        message,
        "Fix .mira/mcp/mcp.json and run /reload.",
    )
