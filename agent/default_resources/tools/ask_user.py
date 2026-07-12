"""Default ask_user tool for MIRA."""

from __future__ import annotations

from langchain.tools import tool
from langgraph.types import interrupt

ASK_USER_INTERRUPT_TYPE = "ask_user"
ASK_USER_OPEN_OPTION = "Tell MIRA what to do differently"


@tool(
    "ask_user",
    description=(
        "Ask the user to choose between concrete next steps when a specific decision requires their input. "
        "In planning mode, every user-facing question that needs an answer must use ask_user; never ask it "
        "in a normal assistant message. Use this only for meaningful choices that materially affect what MIRA "
        "should do next. "
        "Do not use this for generic follow-ups like 'what can I help you with next?' or when you can "
        "make a reasonable safe assumption. Put only the direct question in `question`; put answer choices "
        "only in `options`, and do not enumerate or repeat the choices inside the question text. Prefer 1-3 "
        "concise, mutually exclusive options of about 2-7 words each. Add '(Recommended)' to the best default "
        "only when there is a real default. If the user asks for a list of many options without asking for "
        "ask_user, answer normally in chat. If the user explicitly asks you to use ask_user with many options, "
        "use ask_user and include every requested option. MIRA numbers choices in the UI, so option strings "
        "should not include numbering; good options: ['test_checkpoint.py', 'test_config.py']; bad options: "
        "['1. test_checkpoint.py', '2. test_config.py']. MIRA will always append the final open-ended option "
        "'Tell MIRA what to do differently'."
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
