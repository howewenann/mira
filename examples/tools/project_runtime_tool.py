# Project-runtime tool. Copy this file into .mira/tools/, configure the Execute
# Environment in /settings, then run /reload. Project-only imports must remain
# inside the function because MIRA first imports this module for discovery.

from mira_tool_api import project_tool


@project_tool
def inspect_csv(path: str) -> str:
    """Summarize a CSV using the selected project environment."""
    import pandas as pd

    dataframe = pd.read_csv(path)
    return dataframe.describe(include="all").to_string()
