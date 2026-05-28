import os
from pathlib import Path

from dotenv import load_dotenv


def _int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if not value:
        return default

    try:
        return int(value)
    except ValueError:
        return default


def load_config(workspace: Path) -> dict:
    load_dotenv()

    return {
        "workspace": str(workspace),
        "lmstudio_model": os.getenv("MIRA_LMSTUDIO_MODEL", "local-model"),
        "lmstudio_base_url": os.getenv("MIRA_LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
        "lmstudio_api_key": os.getenv("MIRA_LMSTUDIO_API_KEY", "lm-studio"),
        "tool_output_chars": _int_env("MIRA_TOOL_OUTPUT_CHARS", 240),
        "session_dir": os.getenv("MIRA_SESSION_DIR", str(workspace / ".mira" / "sessions")),
    }
