"""Workspace configuration and process-runtime loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from config.llm import load_model_registry
from config.settings import load_settings_result
from config.tracing import load_tracing_registry


def _int_env(name: str, default: int) -> int:
    """Read an integer environment variable, falling back on invalid input."""
    value = os.getenv(name)
    if not value:
        return default

    try:
        return int(value)
    except ValueError:
        return default


def load_config(workspace: Path, *, override_dotenv: bool = False) -> dict[str, Any]:
    """Load settings, the model registry, and process-scoped runtime values."""
    dotenv_path = workspace / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path, override=override_dotenv)
    else:
        load_dotenv(override=override_dotenv)

    settings_result = load_settings_result(workspace)
    registry = load_model_registry(workspace, environ=os.environ)
    tracing_registry = load_tracing_registry(workspace)
    return {
        "workspace": str(workspace),
        "settings": settings_result.settings,
        "settings_valid": settings_result.valid,
        "model_registry": registry,
        "tracing_registry": tracing_registry,
        "issues": [*settings_result.issues, *registry.issues, *tracing_registry.issues],
        "tool_output_chars": _int_env("MIRA_TOOL_OUTPUT_CHARS", 240),
        "lmstudio_metadata_timeout": _float_env("MIRA_LMSTUDIO_METADATA_TIMEOUT", 2.0),
        "session_dir": os.getenv("MIRA_SESSION_DIR", str(workspace / ".mira" / "_sessions")),
    }


def _float_env(name: str, default: float) -> float:
    """Read a float environment variable, falling back on invalid input."""
    value = os.getenv(name)
    if not value:
        return default

    try:
        return float(value)
    except ValueError:
        return default
