"""Strict read-only authoring validation for Agent Skills."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any

import yaml
from langchain_core.tools import BaseTool, tool

VALIDATE_SKILL_TOOL = "validate_skill"
ALLOWED_FRONTMATTER = {
    "name",
    "description",
    "license",
    "compatibility",
    "allowed-tools",
    "metadata",
}
_FRONTMATTER = re.compile(r"^---\s*\r?\n(.*?)\r?\n---\s*(?:\r?\n|$)", re.DOTALL)


def build_validate_skill_tool(backend: Any) -> BaseTool:
    """Bind the validator to MIRA's read-only workspace backend."""

    @tool(VALIDATE_SKILL_TOOL)
    def validate_skill(path: str) -> dict[str, Any]:
        """Validate a skill directory or SKILL.md path without changing files."""
        skill_path = normalized_skill_path(path)
        response = backend.download_files([skill_path])[0]
        if response.error or response.content is None:
            return {"valid": False, "errors": [f"SKILL.md not found: {skill_path}"]}
        try:
            content = response.content.decode("utf-8")
        except UnicodeDecodeError:
            return {"valid": False, "errors": ["SKILL.md must be UTF-8"]}
        return validate_skill_content(content, PurePosixPath(skill_path).parent.name)

    return validate_skill


def normalized_skill_path(path: str) -> str:
    value = str(path or "").strip().replace("\\", "/").rstrip("/")
    if not value:
        return "/SKILL.md"
    if not value.startswith("/"):
        value = f"/{value}"
    return value if PurePosixPath(value).name == "SKILL.md" else f"{value}/SKILL.md"


def validate_skill_content(content: str, directory_name: str) -> dict[str, Any]:
    """Apply strict dcode-compatible authoring checks to raw SKILL.md text."""
    match = _FRONTMATTER.match(content)
    if match is None:
        return {"valid": False, "errors": ["No valid YAML frontmatter found"]}
    try:
        frontmatter = yaml.safe_load(match.group(1))
    except yaml.YAMLError as error:
        return {"valid": False, "errors": [f"Invalid YAML in frontmatter: {error}"]}
    if not isinstance(frontmatter, dict):
        return {"valid": False, "errors": ["Frontmatter must be a YAML mapping"]}

    errors: list[str] = []
    unexpected = sorted(str(key) for key in frontmatter if key not in ALLOWED_FRONTMATTER)
    if unexpected:
        errors.append(f"Unexpected frontmatter fields: {', '.join(unexpected)}")

    name = frontmatter.get("name")
    if name is None:
        errors.append("Missing required 'name' in frontmatter")
    elif not isinstance(name, str):
        errors.append(f"Name must be a string, got {type(name).__name__}")
    else:
        name = name.strip()
        if not name:
            errors.append("Name cannot be empty")
        else:
            if len(name) > 64:
                errors.append(f"Name is too long ({len(name)} characters); maximum is 64")
            if not valid_skill_name(name):
                errors.append(
                    "Name must use lowercase letters, digits, and single hyphens; "
                    "it cannot start or end with a hyphen"
                )
            if name != directory_name:
                errors.append(f"Frontmatter name '{name}' must match parent directory '{directory_name}'")

    description = frontmatter.get("description")
    if description is None:
        errors.append("Missing required 'description' in frontmatter")
    elif not isinstance(description, str):
        errors.append(f"Description must be a string, got {type(description).__name__}")
    else:
        description = description.strip()
        if not description:
            errors.append("Description cannot be empty")
        if "<" in description or ">" in description:
            errors.append("Description cannot contain angle brackets (< or >)")
        if len(description) > 1024:
            errors.append(f"Description is too long ({len(description)} characters); maximum is 1024")

    compatibility = frontmatter.get("compatibility")
    if compatibility is not None:
        if not isinstance(compatibility, str):
            errors.append(f"Compatibility must be a string, got {type(compatibility).__name__}")
        elif len(compatibility.strip()) > 500:
            errors.append(
                f"Compatibility is too long ({len(compatibility.strip())} characters); maximum is 500"
            )

    return {"valid": not errors, "errors": errors}


def valid_skill_name(name: str) -> bool:
    if name.startswith("-") or name.endswith("-") or "--" in name:
        return False
    return all(
        character == "-"
        or (character.isalpha() and character.islower())
        or character.isdigit()
        for character in name
    )
