"""Translate a skill slash command into a native DeepAgents pin."""

from __future__ import annotations

from dataclasses import dataclass
from langchain_core.messages import BaseMessage, HumanMessage

from agent.skills.registry import SkillRegistry


@dataclass(frozen=True, slots=True)
class PreparedSkill:
    display_text: str
    messages: list[BaseMessage]
    pinned_skills: list[str]


def prepare_skill(invocation: str, registry: SkillRegistry) -> PreparedSkill | None:
    """Pin the discovered skill and pass command text as an ordinary request."""
    resolved = registry.resolve(invocation)
    if resolved is None:
        return None

    return PreparedSkill(
        display_text=invocation,
        messages=[HumanMessage(content=resolved.args)],
        pinned_skills=[resolved.metadata["name"]],
    )
