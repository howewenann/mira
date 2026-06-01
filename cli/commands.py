"""CLI command execution and application bootstrap."""

from __future__ import annotations

import asyncio
import warnings
from pathlib import Path
from typing import Any


def run(prompt: str | None, resume: bool, workspace: Path, session: str | None) -> None:
    """Bridge Typer's synchronous command callback into the async app."""
    from config.llm import ConfigError

    try:
        _suppress_known_warnings()
        asyncio.run(_run(prompt=prompt, resume=resume, workspace=workspace, session=session))
    except ConfigError as error:
        import typer

        typer.echo(f"Configuration error: {error}", err=True)
        raise typer.Exit(code=2) from error


def _suppress_known_warnings() -> None:
    """Hide expected LangChain beta warnings so the CLI stays readable."""
    from langchain_core._api import LangChainBetaWarning

    warnings.filterwarnings("ignore", category=LangChainBetaWarning)


async def _run(prompt: str | None, resume: bool, workspace: Path, session: str | None) -> None:
    """Create the app objects, then run either one-shot or REPL mode."""
    import typer

    from cli.git_guard import ensure_git_repository
    from config.loader import load_config
    from runtime.runner import run_turn
    from ui.renderer import Renderer
    from ui.repl import start_repl

    workspace = workspace.expanduser().resolve()
    config = load_config(workspace)
    renderer = Renderer(tool_output_chars=config["tool_output_chars"])

    if not await ensure_git_repository(workspace, renderer):
        raise typer.Exit(code=1)

    app = _bootstrap(workspace=workspace, session=session, resume=resume, config=config, renderer=renderer)

    if prompt:
        await run_turn(
            agent=app["agent"],
            text=prompt,
            renderer=app["renderer"],
            thread_id=app["session"]["id"],
        )
        app["store"].save(app["session"])
        return

    await start_repl(
        agent=app["agent"],
        plan_agent=app["plan_agent"],
        renderer=app["renderer"],
        store=app["store"],
        session=app["session"],
        model_name=app["model_name"],
    )


def _bootstrap(
    workspace: Path,
    session: str | None,
    resume: bool,
    config: dict[str, Any] | None = None,
    renderer: Any | None = None,
) -> dict[str, Any]:
    """Build config, persistence, renderer, and both action/planning agents."""
    from agent.factory import build_agent, build_plan_agent
    from agent.llm import get_model_name
    from config.loader import load_config
    from session.checkpoint import make_checkpointer
    from session.store import SessionStore
    from ui.renderer import Renderer

    workspace = workspace.expanduser().resolve()
    if config is None:
        config = load_config(workspace)
    if renderer is None:
        renderer = Renderer(tool_output_chars=config["tool_output_chars"])
    store = SessionStore(Path(config["session_dir"]))
    record = store.load(session, resume=resume, workspace=workspace)
    checkpointer = make_checkpointer()
    agent = build_agent(config=config, workspace=workspace, checkpointer=checkpointer)
    plan_agent = build_plan_agent(config=config, workspace=workspace, checkpointer=checkpointer)

    return {
        "agent": agent,
        "plan_agent": plan_agent,
        "config": config,
        "model_name": get_model_name(config),
        "renderer": renderer,
        "session": record,
        "store": store,
    }
