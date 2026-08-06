"""Strict project MCP configuration loading and safe approval previews."""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from agent.mcp.models import MCPConfigIssue, MCPServerState
from agent.resources.paths import MCP_DIR, PROJECT_DIR

MCP_FILE = "mcp.json"
_ENV_REFERENCE = re.compile(r"(\$?)\$\{env:([^}]*)\}")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


@dataclass(frozen=True, slots=True)
class MCPConfiguration:
    exists: bool
    valid: bool
    servers: dict[str, MCPServerState]
    issue: MCPConfigIssue | None = None


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
        return MCPConfiguration(True, False, {}, MCPConfigIssue(_error_text(error)))
    if not isinstance(raw, dict) or not isinstance(raw.get("mcpServers"), dict):
        return MCPConfiguration(
            True,
            False,
            {},
            MCPConfigIssue("top level must be an object containing an mcpServers object"),
        )

    servers: dict[str, MCPServerState] = {}
    for alias, definition in raw["mcpServers"].items():
        name = str(alias) if isinstance(alias, str) else ""
        if not name.strip():
            continue
        try:
            transport, effective = normalize_server_definition(definition)
            runtime_config = resolve_environment_references(effective, environ=environ)
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
        servers[name] = state
    return MCPConfiguration(True, True, servers)


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


def resolve_environment_references(
    value: Any,
    *,
    environ: Mapping[str, str] | None = None,
) -> Any:
    """Resolve explicit ``${env:NAME}`` references in JSON values once."""
    source = os.environ if environ is None else environ
    if isinstance(value, str):
        unmatched = _ENV_REFERENCE.sub("", value)
        if "${env:" in unmatched:
            raise ValueError("invalid environment reference; use ${env:NAME}")

        def replace(match: re.Match[str]) -> str:
            if match.group(1):
                return match.group(0)[1:]
            name = match.group(2)
            if _ENV_NAME.fullmatch(name) is None:
                raise ValueError(
                    f"invalid environment variable name {name!r}; use ${{env:NAME}}"
                )
            if name not in source:
                raise ValueError(
                    f"environment variable {name} is not set; define it before starting MIRA "
                    "or in the workspace .env"
                )
            return source[name]

        return _ENV_REFERENCE.sub(replace, value)
    if isinstance(value, list):
        return [resolve_environment_references(item, environ=source) for item in value]
    if isinstance(value, dict):
        return {
            key: resolve_environment_references(item, environ=source)
            for key, item in value.items()
        }
    return value


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
