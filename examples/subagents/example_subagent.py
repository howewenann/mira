"""Example project subagent to copy into ``.mira/subagents/``."""

SUBAGENTS = [
    {
        "name": "example-project-guide",
        "description": "Answers questions using this project's conventions.",
        "system_prompt": (
            "Help with this project. Follow its documented commands, architecture, "
            "and conventions, and call out any missing context."
        ),
    }
]
