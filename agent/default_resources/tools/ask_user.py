"""Default ask_user tool for MIRA."""

from __future__ import annotations

from langchain.tools import tool
from langgraph.types import interrupt

ASK_USER_INTERRUPT_TYPE = "ask_user"
ASK_USER_OPEN_OPTION = "Tell MIRA what to do differently"


@tool(
    "ask_user",
    description=(
        "Ask the user to choose between concrete next steps when you are blocked on a specific decision. "
        "Use this only for meaningful choices that materially affect what MIRA should do next. "
        "Do not use this for generic follow-ups like 'what can I help you with next?' or when you can "
        "make a reasonable safe assumption. Provide concise, mutually exclusive options; MIRA will always "
        "append the final open-ended option 'Tell MIRA what to do differently'."
    ),
)
def ask_user(question: str, options: list[str]) -> str:
    """Pause and ask the user to choose a concrete next step."""
    return str(
        interrupt(
            {
                "type": ASK_USER_INTERRUPT_TYPE,
                "question": question,
                "options": normalize_options(options),
                "open_option": ASK_USER_OPEN_OPTION,
            }
        )
    )


def normalize_options(options: list[str]) -> list[str]:
    """Return unique non-empty options with the open-ended choice last."""
    normalized = []
    seen = set()
    for option in options:
        text = " ".join(str(option).split())
        if not text or text == ASK_USER_OPEN_OPTION or text in seen:
            continue
        normalized.append(text)
        seen.add(text)

    normalized.append(ASK_USER_OPEN_OPTION)
    return normalized
