"""Process-local launch options and effective runtime configuration."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit

from config import loader


@dataclass(frozen=True, slots=True)
class LaunchOptions:
    """Immutable options supplied when the current MIRA process was launched."""

    llm_direct: bool = False


@dataclass(frozen=True, slots=True)
class RuntimeSnapshot:
    """Sanitized description of the runtime currently used by active agents."""

    model_name: str
    provider: str | None
    endpoint: str | None
    direct_effective: bool
    direct_requested: bool
    warnings: tuple[str, ...] = ()


def load_effective_config(
    workspace: Path,
    launch_options: LaunchOptions,
    *,
    override_dotenv: bool = False,
) -> dict[str, Any]:
    """Load reloadable configuration and overlay process-local launch options."""
    if override_dotenv:
        config = dict(loader.load_config(workspace, override_dotenv=True))
    else:
        config = dict(loader.load_config(workspace))
    config["llm_direct"] = launch_options.llm_direct
    return config


def build_runtime_snapshot(
    effective_config: Mapping[str, Any],
    launch_options: LaunchOptions,
    *,
    model_name: str,
) -> RuntimeSnapshot:
    """Build a sanitized snapshot from the config used to create active agents."""
    direct_effective = bool(effective_config.get("llm_direct"))
    direct_requested = launch_options.llm_direct
    warnings = ()
    if direct_requested and not direct_effective:
        warnings = ("Direct mode was requested but is not present in the effective configuration.",)
    elif direct_effective and not direct_requested:
        warnings = ("Direct mode is effective even though it was not requested at launch.",)

    provider_value = effective_config.get("llm_provider")
    provider = provider_value.strip() if isinstance(provider_value, str) and provider_value.strip() else None
    return RuntimeSnapshot(
        model_name=model_name.strip() or "unknown",
        provider=provider,
        endpoint=_sanitize_endpoint(effective_config.get("llm_base_url")),
        direct_effective=direct_effective,
        direct_requested=direct_requested,
        warnings=warnings,
    )


def _sanitize_endpoint(value: Any) -> str | None:
    """Return a display-safe endpoint without credentials, query, or fragment."""
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = urlsplit(value.strip())
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return None
    if not parsed.scheme or not hostname:
        return None

    display_host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = f"{display_host}:{port}" if port is not None else display_host
    return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
