# Standard MIRA-runtime tool. Copy this file into .mira/tools/, edit it, then
# run /reload. Imported packages must be installed in MIRA's environment.

from langchain_core.tools import tool


@tool
def count_words(text: str) -> int:
    """Count the number of words in text."""
    return len(text.split())
