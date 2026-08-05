"""Parse local-file references from visible user text."""

from __future__ import annotations

import re


_REFERENCE_PATTERN = re.compile(r'(?<![\w@])@(?:"([^"\r\n]+)"|([^\s@]+))')
_TRAILING_PUNCTUATION = ".,;:!?)]}"


def local_file_references(text: str) -> list[str]:
    """Return normalized, ordered, exactly deduplicated virtual file paths."""
    references: list[str] = []
    seen: set[str] = set()
    for match in _REFERENCE_PATTERN.finditer(text):
        quoted, plain = match.groups()
        value = quoted if quoted is not None else _trim_plain_reference(plain or "")
        normalized = normalize_virtual_file_path(value)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        references.append(normalized)
    return references


def normalize_virtual_file_path(path: str) -> str:
    """Normalize a mention to the absolute virtual path used by DeepAgents."""
    value = path.strip().replace("\\", "/")
    while value.startswith("./"):
        value = value[2:]
    value = value.lstrip("/")
    return f"/{value}" if value else ""


def _trim_plain_reference(value: str) -> str:
    return value.rstrip(_TRAILING_PUNCTUATION)


__all__ = ["local_file_references", "normalize_virtual_file_path"]
