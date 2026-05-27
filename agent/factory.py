from pathlib import Path

from deepagents import FilesystemPermission, create_deep_agent
from deepagents.backends import FilesystemBackend
from deepagents.middleware.summarization import create_summarization_tool_middleware
from langchain_quickjs import CodeInterpreterMiddleware

from agent.llm import get_llm

SKILLS = []
MEMORY = []
SUBAGENTS = []


def build_agent(config: dict, workspace: Path, checkpointer):
    model = get_llm(config)
    backend = FilesystemBackend(root_dir=str(workspace), virtual_mode=True)

    permissions = [
        FilesystemPermission(
            operations=["read", "write"],
            paths=["/**"],
            mode="allow",
        ),
    ]

    middleware = [
        CodeInterpreterMiddleware(),
        create_summarization_tool_middleware(model=model, backend=backend),
    ]

    return create_deep_agent(
        model=model,
        backend=backend,
        middleware=middleware,
        skills=SKILLS,
        memory=MEMORY,
        subagents=SUBAGENTS,
        permissions=permissions,
        interrupt_on={
            "write_file": {"allowed_decisions": ["approve", "edit", "reject"]},
            "edit_file": {"allowed_decisions": ["approve", "edit", "reject"]},
        },
        checkpointer=checkpointer,
    )
