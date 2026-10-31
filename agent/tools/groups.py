"""The configurable built-in hierarchy shared by Settings and Access."""

from agent.tools.skill_validation import VALIDATE_SKILL_METADATA, VALIDATE_SKILL_TOOL
from config.settings import CORE_READ_ONLY_TOOLS, INBUILT_DANGEROUS_TOOLS


def builtin_tool_groups(
    metadata: list[dict[str, str]],
) -> list[tuple[str, list[tuple[str, tuple[str, ...]]]]]:
    """Core follows policy order; bundled features opt in with TOOL_GROUP."""
    core_names = {*CORE_READ_ONLY_TOOLS, *INBUILT_DANGEROUS_TOOLS}
    mira = {
        item["name"]: item for item in metadata
        if item.get("group") and item.get("name") and item.get("source") in {"default", "project"}
    }
    mira[VALIDATE_SKILL_TOOL] = VALIDATE_SKILL_METADATA
    features: dict[str, set[str]] = {}
    for name, item in mira.items():
        if name not in core_names:
            features.setdefault(item["group"], set()).add(name)
    return [
        ("Core Tools", [("Read-only", CORE_READ_ONLY_TOOLS), ("Actions", INBUILT_DANGEROUS_TOOLS)]),
        ("MIRA Tools", [
            (group, tuple(sorted(names, key=str.casefold)))
            for group, names in sorted(features.items(), key=lambda item: item[0].casefold())
        ]),
    ]
