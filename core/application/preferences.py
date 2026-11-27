"""Detached configuration and offline editing before a runtime can start."""

from copy import deepcopy
from pathlib import Path
from typing import Any


def configuration_view(config: dict[str, Any]) -> dict[str, Any]:
    """Keep credentials, model clients and runtime resources out of UI metadata."""
    return {
        "settings": deepcopy(config.get("settings") or {}),
        "model_aliases": tuple(getattr(config.get("model_registry"), "profiles", {})),
        "tracing_profiles": tuple(getattr(config.get("tracing_registry"), "profiles", {})),
        "issues": deepcopy(list(config.get("issues") or [])),
    }


def read_configuration(workspace: Path | str) -> dict[str, Any]:
    """Read editable workspace preferences without constructing agents or MCP."""
    from config.loader import load_config

    return configuration_view(load_config(Path(workspace)))


def configure_workspace(workspace: Path | str, settings: dict[str, Any]) -> tuple[bool, str]:
    """Save preferences for startup recovery; no active runtime is changed."""
    from config.settings import normalize_settings, save_settings

    if not save_settings(Path(workspace), normalize_settings(deepcopy(settings))):
        return False, "could not save .mira/settings.yml"
    return True, "settings saved; Reload Runtime required"


def configure_git(workspace: Path | str, preference: bool) -> tuple[bool, str, dict[str, Any]]:
    """Apply Git protection even when agent startup has failed."""
    from config.settings import load_settings, save_settings, set_git_protection
    from core.workspace import init_git_repository, is_git_worktree

    workspace = Path(workspace)
    settings = load_settings(workspace)
    if preference and not is_git_worktree(workspace) and not init_git_repository(workspace):
        return False, "Git could not initialize; protection remains unconfigured", settings
    updated = set_git_protection(settings, preference)
    if not save_settings(workspace, updated):
        return False, "could not save .mira/settings.yml", settings
    message = "Git protection enabled; repository ready" if preference else "Git protection declined; no reload required"
    return True, message, updated


def preview_tracing(workspace: Path | str, settings: dict[str, Any]) -> str:
    """Read a credential-redacted preview without starting agents or transports."""
    from config.tracing import load_tracing_registry

    return _tracing_preview(settings, load_tracing_registry(Path(workspace)))


def _tracing_preview(settings: dict[str, Any], registry: Any) -> str:
    from config.settings import tracing_settings
    from tracing.bootstrap import tracing_yaml_fragment

    values = tracing_settings(settings)
    selected = str(values["profile"])
    profile = registry.profile(selected)
    if profile is None:
        return f"Preview unavailable: tracing profile '{selected}' was not found"
    return tracing_yaml_fragment(enabled=bool(values["enabled"]), profile=selected,
        middleware_spans=str(values["middleware_spans"]), endpoint=profile.endpoint,
        headers=profile.headers, span_attributes=profile.span_attributes)
