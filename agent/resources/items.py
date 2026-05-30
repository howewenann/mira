"""Small helpers for resource display metadata."""

from __future__ import annotations

from typing import Any


def merge_project_overrides(
    defaults: list[dict[str, Any]],
    projects: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = {item["name"]: dict(item) for item in defaults}
    default_names = set(merged)

    for item in projects:
        merged[item["name"]] = {
            **item,
            "replaces": "default" if item["name"] in default_names else "",
        }

    return list(merged.values())


def display_item(item: dict[str, Any]) -> dict[str, str]:
    return {
        "name": str(item["name"]),
        "path": str(item["path"]),
        "source": str(item["source"]),
        "replaces": str(item.get("replaces") or ""),
    }
