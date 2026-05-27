import os
from pathlib import Path

from dotenv import load_dotenv


def load_config(workspace: Path) -> dict:
    load_dotenv()

    return {
        "workspace": str(workspace),
        "model": os.getenv("MIRA_MODEL", ""),
        "lmstudio_model": os.getenv("MIRA_LMSTUDIO_MODEL", "local-model"),
        "lmstudio_base_url": os.getenv("MIRA_LMSTUDIO_BASE_URL", "http://localhost:1234/v1"),
        "lmstudio_api_key": os.getenv("MIRA_LMSTUDIO_API_KEY", "lm-studio"),
        "session_dir": os.getenv("MIRA_SESSION_DIR", str(workspace / ".mira" / "sessions")),
    }
