"""Acceptance-criteria generation outside the main agent loop."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from agent.llm import get_llm
from config.metadata import ModelMetadata

INITIAL_CRITERIA_PROMPT = """You draft acceptance criteria for a general-purpose agent objective.

Return only a concise Markdown bullet list that the user can review before work begins.

Each criterion should:
- describe a concrete and observable outcome;
- be framed as a condition that must be true for the objective to be complete;
- reflect the user's requested deliverables, constraints, and intended result;
- include appropriate evidence or verification when relevant;
- distinguish successful completion from a genuine blocker requiring information, access, credentials, or a decision from the user.

Adapt the criteria to the task. The objective may involve research, analysis, writing, planning, coding, file operations, investigation, communication, or other work.

Do not create an execution plan.
Do not perform the task.
Do not add requirements unsupported by the user's objective.
Do not include introductory or concluding prose.
Return only the complete Markdown bullet list."""

REVISION_CRITERIA_PROMPT = """You are revising only the acceptance criteria. You are not producing or revising an execution plan.

The user's feedback may mention "the plan", "steps", "method", or "approach" because the same feedback will later be passed to a separate plan-revision process.

Update the acceptance criteria only when the feedback changes the required outcome, deliverables, constraints, scope, or conditions for completion.

If the feedback only changes the plan's wording, ordering, level of detail, or execution approach, return the previous acceptance criteria exactly unchanged.

Preserve criteria that remain valid.
Do not add scope unsupported by the original objective or the user's feedback.
Return the complete revised Markdown bullet list.
Do not output a plan or explanatory prose."""


class GoalCriteriaService:
    """Generate and revise Definition-of-Done Markdown with MIRA's active model."""

    def __init__(self, config: dict[str, Any], metadata: ModelMetadata | None = None) -> None:
        self.config = config
        self.metadata = metadata

    async def generate(self, objective: str) -> str:
        """Generate initial criteria for an effective objective."""
        return await self._invoke(
            INITIAL_CRITERIA_PROMPT,
            f"<objective>\n{objective.strip()}\n</objective>",
        )

    async def revise(self, objective: str, previous_criteria: str, feedback: str) -> str:
        """Revise criteria without accepting or exposing a previous plan."""
        return await self._invoke(
            REVISION_CRITERIA_PROMPT,
            "\n\n".join(
                (
                    f"<objective>\n{objective.strip()}\n</objective>",
                    f"<previous_criteria>\n{previous_criteria.strip()}\n</previous_criteria>",
                    f"<user_feedback>\n{feedback.strip()}\n</user_feedback>",
                )
            ),
        )

    async def _invoke(self, system_prompt: str, user_prompt: str) -> str:
        model = get_llm(self.config, metadata=self.metadata)
        response = await model.ainvoke(
            [SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)]
        )
        text = response_text(response).strip()
        if not text:
            raise RuntimeError("criteria model returned an empty response")
        return text


def response_text(response: Any) -> str:
    """Return visible text from common LangChain response content shapes."""
    content = getattr(response, "content", response)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                parts.append(str(block.get("text") or ""))
        return "".join(parts)
    return str(content or "")
