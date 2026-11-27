"""Normalization and display text for deterministic correction events."""

from __future__ import annotations

from typing import Any

from agent.middleware.correction import CORRECTION_EVENT


def normalize_correction_event(event: dict[str, Any]) -> dict[str, Any]:
    """Return a stable renderer/session representation."""
    return {
        "type": CORRECTION_EVENT,
        "protocol": str(event.get("protocol") or "correction").strip(),
        "check_name": str(event.get("check_name") or "Correction").strip(),
        "workflow": str(event.get("workflow") or "Correction").strip(),
        "failed_check": str(event.get("failed_check") or "The response check failed.").strip(),
        "retry_prompt": str(event.get("retry_prompt") or "").strip(),
        "attempt": max(0, int(event.get("attempt") or 0)),
        "max_retries": max(1, int(event.get("max_retries") or 1)),
        "exhausted": event.get("exhausted") is True,
        "terminal_text": str(event.get("terminal_text") or "").strip(),
    }


def correction_context_text(event: dict[str, Any]) -> str:
    """Return correction context that is clearly not user-authored."""
    value = normalize_correction_event(event)
    lines = [
        f"{value['check_name']} check ({value['workflow']}) failed: "
        f"{value['failed_check']}"
    ]
    if value["exhausted"]:
        lines.append(f"Retry limit reached after {value['max_retries']} retries.")
    elif value["retry_prompt"]:
        lines.append(f"Correction prompt: {value['retry_prompt']}")
    return "\n".join(lines)


__all__ = [
    "correction_context_text",
    "normalize_correction_event",
]
