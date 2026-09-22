# Project-runtime tool. Copy this file into .mira/tools/, configure the Execute
# Environment in /settings, then run /reload. Project-only imports must remain
# inside the function because MIRA first imports this module for discovery.

from pathlib import Path
from tempfile import TemporaryDirectory

from mira_tool_api import project_tool


@project_tool
def inspect_csv(path: str) -> str:
    """Summarize a CSV using the selected project environment."""
    import pandas as pd

    dataframe = pd.read_csv(path)
    return dataframe.describe(include="all").to_string()


def main() -> None:
    with TemporaryDirectory() as directory:
        csv_path = Path(directory) / "example.csv"
        csv_path.write_text("name,score\nalice,10\nbob,20\n", encoding="utf-8")

        # Direct smoke tests call the plain Python function normally. Execution
        # through MIRA uses the configured project-runtime machinery instead.
        print(inspect_csv(str(csv_path)))


if __name__ == "__main__":
    main()
