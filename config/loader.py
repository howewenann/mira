"""Environment-based configuration loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from config.llm import load_llm_config
from config.settings import load_settings


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
    """Load all runtime configuration from the environment and defaults."""
    dotenv_path = workspace / ".env"
    if dotenv_path.exists():
        load_dotenv(dotenv_path, override=override_dotenv)
    else:
        load_dotenv(override=override_dotenv)

    return {
        "workspace": str(workspace),
        "settings": load_settings(workspace),
        **load_llm_config(os.environ),
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
