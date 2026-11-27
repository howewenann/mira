"""Correction bubble text for normalized frontend event payloads."""

from typing import Any

def correction_title(event: dict[str, Any]) -> str:
    """Return the compact visible correction label."""
    check_name = str(event.get("check_name") or "Correction")
    return f"{check_name} check"


def correction_text(event: dict[str, Any]) -> str:
    """Return simple technical details for one correction bubble."""
    value = event
    lines = [f"Workflow: {value['workflow']}", f"Check failed: {value['failed_check']}"]
    if value.get("exhausted", False):
        lines.append(f"Retry limit reached: {value.get('max_retries', 1)} of {value.get('max_retries', 1)}")
    else:
        lines.append(f"Retry prompt: {value.get('retry_prompt', '')}")
        lines.append(f"Retry {value.get('attempt', 0)} of {value.get('max_retries', 1)}")
    return "\n".join(lines)
