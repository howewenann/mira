"""Workspace settings stored under .mira/settings.yml."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

SETTINGS_FILE = "settings.yml"
EXECUTE_TOOL = "execute"
LOCKED_INBUILT_DANGEROUS_TOOLS = ("write_file", "edit_file", "eval", "task")
INBUILT_DANGEROUS_TOOLS = (*LOCKED_INBUILT_DANGEROUS_TOOLS, EXECUTE_TOOL)
DEFAULT_APPROVAL_TOOLS = INBUILT_DANGEROUS_TOOLS
DEFAULT_SETTINGS: dict[str, Any] = {
    "hitl": {
        "git_protection": {"enabled": True},
        "tools": {
            "write_file": {"enabled": True, "always_allow": False},
            "edit_file": {"enabled": True, "always_allow": False},
            "eval": {"enabled": True, "always_allow": False},
            "task": {"enabled": True, "always_allow": False},
            "execute": {"enabled": False, "always_allow": False},
        },
    }
}


def settings_path(workspace: Path) -> Path:
    """Return the workspace-local settings path."""
    return workspace.expanduser().resolve() / ".mira" / SETTINGS_FILE


def load_settings(workspace: Path) -> dict[str, Any]:
    """Load normalized workspace settings, falling back to defaults."""
    path = settings_path(workspace)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError):
        raw = {}
    return normalize_settings(raw)


def save_settings(workspace: Path, settings: dict[str, Any]) -> bool:
    """Persist normalized workspace settings."""
    path = settings_path(workspace)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            yaml.safe_dump(normalize_settings(settings), sort_keys=False),
            encoding="utf-8",
        )
    except OSError:
        return False
    return True


def normalize_settings(raw: Any) -> dict[str, Any]:
    """Return settings with defaults and only supported HITL shapes."""
    settings = deepcopy(DEFAULT_SETTINGS)
    if not isinstance(raw, dict):
        return settings

    hitl = raw.get("hitl")
    if not isinstance(hitl, dict):
        return settings

    git_protection = hitl.get("git_protection")
    if isinstance(git_protection, dict) and isinstance(git_protection.get("enabled"), bool):
        settings["hitl"]["git_protection"]["enabled"] = git_protection["enabled"]

    tools = hitl.get("tools")
    if isinstance(tools, dict):
        normalized_tools = {name: dict(spec) for name, spec in settings["hitl"]["tools"].items()}
        for name, spec in tools.items():
            if not isinstance(name, str) or not name.strip() or not isinstance(spec, dict):
                continue
            always_allow = spec.get("always_allow")
            enabled = spec.get("enabled")
            current = dict(normalized_tools.get(name, {"enabled": True, "always_allow": False}))
            if isinstance(enabled, bool):
                current["enabled"] = enabled
            if isinstance(always_allow, bool):
                current["always_allow"] = always_allow
            if name in LOCKED_INBUILT_DANGEROUS_TOOLS:
                current["enabled"] = True
            normalized_tools[name] = current
        settings["hitl"]["tools"] = normalized_tools

    return settings


def hitl_settings(config_or_settings: dict[str, Any] | None) -> dict[str, Any]:
    """Extract the HITL section from a runtime config or settings object."""
    if not isinstance(config_or_settings, dict):
        return deepcopy(DEFAULT_SETTINGS["hitl"])
    settings = config_or_settings.get("settings", config_or_settings)
    return normalize_settings(settings).get("hitl", deepcopy(DEFAULT_SETTINGS["hitl"]))


def git_protection_enabled(config_or_settings: dict[str, Any] | None) -> bool:
    """Return whether startup Git protection is enabled."""
    hitl = hitl_settings(config_or_settings)
    return bool(hitl.get("git_protection", {}).get("enabled", True))


def tool_always_allow(config_or_settings: dict[str, Any] | None, tool_name: str) -> bool:
    """Return whether a tool is configured to skip HITL approval."""
    hitl = hitl_settings(config_or_settings)
    tools = hitl.get("tools", {})
    spec = tools.get(tool_name) if isinstance(tools, dict) else None
    if isinstance(spec, dict) and isinstance(spec.get("always_allow"), bool):
        return bool(spec["always_allow"])
    return tool_name not in DEFAULT_APPROVAL_TOOLS


def tool_enabled(config_or_settings: dict[str, Any] | None, tool_name: str) -> bool:
    """Return whether a configurable user tool is enabled."""
    if tool_name in LOCKED_INBUILT_DANGEROUS_TOOLS:
        return True
    hitl = hitl_settings(config_or_settings)
    tools = hitl.get("tools", {})
    spec = tools.get(tool_name) if isinstance(tools, dict) else None
    if isinstance(spec, dict) and isinstance(spec.get("enabled"), bool):
        return bool(spec["enabled"])
    return True


def set_git_protection(settings: dict[str, Any], enabled: bool) -> dict[str, Any]:
    """Return settings with the Git protection toggle updated."""
    updated = normalize_settings(settings)
    updated["hitl"]["git_protection"]["enabled"] = bool(enabled)
    return updated


def set_tool_always_allow(settings: dict[str, Any], tool_name: str, always_allow: bool) -> dict[str, Any]:
    """Return settings with one tool approval toggle updated."""
    updated = normalize_settings(settings)
    current = dict(updated["hitl"].setdefault("tools", {}).get(tool_name, {"enabled": True}))
    current["always_allow"] = bool(always_allow)
    if tool_name in LOCKED_INBUILT_DANGEROUS_TOOLS:
        current["enabled"] = True
    updated["hitl"]["tools"][tool_name] = current
    return updated


def set_tool_enabled(settings: dict[str, Any], tool_name: str, enabled: bool) -> dict[str, Any]:
    """Return settings with one configurable tool enabled or disabled."""
    updated = normalize_settings(settings)
    current = dict(updated["hitl"].setdefault("tools", {}).get(tool_name, {"always_allow": False}))
    current["enabled"] = True if tool_name in LOCKED_INBUILT_DANGEROUS_TOOLS else bool(enabled)
    current.setdefault("always_allow", False)
    updated["hitl"]["tools"][tool_name] = current
    return updated
