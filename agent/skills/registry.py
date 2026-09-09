"""Registry for deterministic skill slash commands."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass


SKILL_PREFIX = "/skill__"


@dataclass(frozen=True, slots=True)
class ResolvedSkill:
    metadata: Mapping[str, str]
    args: str


class SkillRegistry:
    """Exact command lookup over already-discovered skill metadata."""

    def __init__(self, skills: Iterable[Mapping[str, str]] = ()) -> None:
        self.commands: dict[str, Mapping[str, str]] = {}
        for skill in skills:
            command = f"{SKILL_PREFIX}{skill['name']}"
            if command in self.commands:
                raise ValueError(f"duplicate skill command: {command}")
            self.commands[command] = skill

    def resolve(self, invocation: str) -> ResolvedSkill | None:
        if not invocation.startswith(SKILL_PREFIX):
            return None

        split_at = next(
            (index for index, character in enumerate(invocation) if character.isspace()),
            len(invocation),
        )
        command = invocation[:split_at]
        args = invocation[split_at + 1 :] if split_at < len(invocation) else ""
        if command == SKILL_PREFIX:
            raise ValueError("usage: /skill__<name> [request]")
        skill = self.commands.get(command)
        if skill is None:
            raise ValueError(f"unknown skill command: {command}")
        return ResolvedSkill(skill, args)
