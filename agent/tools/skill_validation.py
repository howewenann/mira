"""Strict read-only authoring validation for Agent Skills."""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import Any

import yaml
from deepagents.middleware.skills import MAX_SKILL_FILE_SIZE
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
        file_errors: list[str] = []
        location_errors = validate_skill_location(skill_path)
        size_errors: list[str] = []
        content_errors: list[str] = []

        response = backend.download_files([skill_path])[0]
        if response.error or response.content is None:
            file_errors.append(f"[FILE] Create a readable SKILL.md file at '{skill_path}'.")
        else:
            try:
                content = response.content.decode("utf-8")
            except UnicodeDecodeError:
                file_errors.append(f"[FILE] Save '{skill_path}' as valid UTF-8 text.")
            else:
                if len(content) > MAX_SKILL_FILE_SIZE:
                    size_errors.append(
                        f"[SIZE] Reduce '{skill_path}' to {MAX_SKILL_FILE_SIZE} characters or fewer; "
                        f"it currently has {len(content)}."
                    )
                directory_name = PurePosixPath(skill_path).parent.name
                content_errors = validate_skill_content(content, directory_name)["errors"]

        errors = [*file_errors, *location_errors, *size_errors, *content_errors]
        return {"valid": not errors, "errors": errors}

    return validate_skill


def normalized_skill_path(path: str) -> str:
    """Normalize separators and resolve a directory argument to its SKILL.md."""
    value = str(path or "").strip().replace("\\", "/").rstrip("/")
    if not value:
        return "/SKILL.md"
    if not value.startswith("/"):
        value = f"/{value}"
    return value if PurePosixPath(value).name == "SKILL.md" else f"{value}/SKILL.md"


def validate_skill_location(skill_path: str) -> list[str]:
    """Require one project skill directory directly below /.mira/skills."""
    path = PurePosixPath(skill_path)
    parts = path.parts
    valid = (
        str(path) == skill_path
        and len(parts) == 5
        and parts[:3] == ("/", ".mira", "skills")
        and parts[3] not in {"", ".", ".."}
        and parts[4] == "SKILL.md"
    )
    if valid:
        return []

    directory_name = path.parent.name
    expected_name = directory_name if directory_name not in {"", ".", ".."} else "<skill-name>"
    expected = f"/.mira/skills/{expected_name}/SKILL.md"
    return [
        f"[LOCATION] Move the skill from '{skill_path}' to '{expected}'; "
        "skills must be directly under '/.mira/skills'."
    ]


def validate_skill_content(content: str, directory_name: str) -> dict[str, Any]:
    """Apply strict dcode-compatible authoring checks to raw SKILL.md text."""
    frontmatter_errors: list[str] = []
    name_errors: list[str] = []
    description_errors: list[str] = []
    compatibility_errors: list[str] = []
    body_errors: list[str] = []

    match = _FRONTMATTER.match(content)
    if match is None:
        frontmatter_errors.append(
            "[FRONTMATTER] Add valid YAML frontmatter enclosed by opening and closing '---' delimiters."
        )
        return {"valid": False, "errors": frontmatter_errors}

    if not content[match.end() :].strip():
        body_errors.append("[BODY] Add Markdown instructions after the YAML frontmatter.")

    try:
        frontmatter = yaml.safe_load(match.group(1))
    except yaml.YAMLError:
        frontmatter_errors.append(
            "[FRONTMATTER] Fix invalid YAML between the opening and closing '---' delimiters."
        )
        errors = [*frontmatter_errors, *body_errors]
        return {"valid": False, "errors": errors}

    if not isinstance(frontmatter, dict):
        frontmatter_errors.append("[FRONTMATTER] Change the YAML frontmatter to a mapping of fields.")
        errors = [*frontmatter_errors, *body_errors]
        return {"valid": False, "errors": errors}

    unexpected = sorted(str(key) for key in frontmatter if key not in ALLOWED_FRONTMATTER)
    if unexpected:
        frontmatter_errors.append(
            f"[FRONTMATTER] Remove unsupported fields: {', '.join(unexpected)}."
        )

    name = frontmatter.get("name")
    if name is None:
        name_errors.append("[NAME] Add the required 'name' field to the frontmatter.")
    elif not isinstance(name, str):
        name_errors.append(f"[NAME] Change 'name' to a string instead of {type(name).__name__}.")
    else:
        name = name.strip()
        if not name:
            name_errors.append("[NAME] Set 'name' to a non-empty skill name.")
        else:
            if len(name) > 64:
                name_errors.append(
                    f"[NAME] Shorten 'name' from {len(name)} characters to 64 or fewer."
                )
            if not valid_skill_name(name):
                name_errors.append(
                    "[NAME] Use only lowercase letters, digits, and single hyphens, "
                    "with no leading or trailing hyphen."
                )
            if name != directory_name:
                name_errors.append(
                    f"[NAME] Change frontmatter name from '{name}' to '{directory_name}' "
                    "so it matches the parent directory."
                )

    description = frontmatter.get("description")
    if description is None:
        description_errors.append(
            "[DESCRIPTION] Add the required 'description' field to the frontmatter."
        )
    elif not isinstance(description, str):
        description_errors.append(
            f"[DESCRIPTION] Change 'description' to a string instead of "
            f"{type(description).__name__}."
        )
    else:
        description = description.strip()
        if not description:
            description_errors.append("[DESCRIPTION] Set 'description' to a non-empty value.")
        if "<" in description or ">" in description:
            description_errors.append(
                "[DESCRIPTION] Remove angle brackets ('<' and '>') from 'description'."
            )
        if len(description) > 1024:
            description_errors.append(
                f"[DESCRIPTION] Shorten 'description' from {len(description)} characters "
                "to 1024 or fewer."
            )

    compatibility = frontmatter.get("compatibility")
    if compatibility is not None:
        if not isinstance(compatibility, str):
            compatibility_errors.append(
                f"[COMPATIBILITY] Change 'compatibility' to a string instead of "
                f"{type(compatibility).__name__}."
            )
        elif len(compatibility.strip()) > 500:
            compatibility_errors.append(
                f"[COMPATIBILITY] Shorten 'compatibility' from "
                f"{len(compatibility.strip())} characters to 500 or fewer."
            )

    errors = [
        *frontmatter_errors,
        *name_errors,
        *description_errors,
        *compatibility_errors,
        *body_errors,
    ]
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
