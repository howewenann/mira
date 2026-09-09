"""Load an exact registered skill into MIRA's prepared-message flow."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage

from agent.skills.registry import SkillRegistry


@dataclass(frozen=True, slots=True)
class PreparedSkill:
    display_text: str
    messages: list[BaseMessage]


def prepare_skill(invocation: str, registry: SkillRegistry, backend: Any) -> PreparedSkill | None:
    """Prepare one explicit skill invocation without rediscovery."""
    resolved = registry.resolve(invocation)
    if resolved is None:
        return None

    skill = resolved.metadata
    response = backend.download_files([skill["path"]])[0]
    if response.error or response.content is None:
        raise ValueError(f"could not read skill '{skill['name']}': {response.error or 'empty response'}")
    try:
        content = response.content.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"could not read skill '{skill['name']}': SKILL.md is not UTF-8") from error
    if not content.strip():
        raise ValueError(f"skill '{skill['name']}' has an empty SKILL.md")

    prompt = (
        f"I'm invoking the skill `{skill['name']}`. "
        "Below are the full instructions from the skill's SKILL.md file. "
        "Follow these instructions to complete the task.\n\n"
        f"---\n{content}\n---"
    )
    if resolved.args:
        prompt += f"\n\n**User request:** {resolved.args}"

    message = HumanMessage(
        content=prompt,
        additional_kwargs={
            "__skill": {
                "name": skill["name"],
                "description": skill.get("description", ""),
                "source": skill.get("source", ""),
                "args": resolved.args,
            }
        },
    )
    return PreparedSkill(display_text=invocation, messages=[message])
