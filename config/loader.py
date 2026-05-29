"""Environment-based configuration loading."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


def _int_env(name: str, default: int) -> int:
    """Read an integer environment variable, falling back on invalid input."""
    value = os.getenv(name)
    if not value:
        return default

    try:
        return int(value)
    except ValueError:
        return default


def load_config(workspace: Path) -> dict[str, Any]:
    """Load all runtime configuration from the environment and defaults."""
    load_dotenv()

    return {
        "workspace": str(workspace),
        "lmstudio_model": os.getenv("MIRA_LMSTUDIO_MODEL", "local-model"),
        "lmstudio_base_url": os.getenv("MIRA_LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
        "lmstudio_api_key": os.getenv("MIRA_LMSTUDIO_API_KEY", "lm-studio"),
        "tool_output_chars": _int_env("MIRA_TOOL_OUTPUT_CHARS", 240),
        "session_dir": os.getenv("MIRA_SESSION_DIR", str(workspace / ".mira" / "sessions")),
    }
