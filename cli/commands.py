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
    from session.context import append_turn, compact_if_needed, update_title_once, will_compact, with_resume_context
    from ui.renderer import Renderer
    from ui.repl import start_repl

    workspace = workspace.expanduser().resolve()
    config = load_config(workspace)
    renderer = Renderer(tool_output_chars=config["tool_output_chars"])

    if not await ensure_git_repository(workspace, renderer):
        raise typer.Exit(code=1)

    app = _bootstrap(workspace=workspace, session=session, resume=resume, config=config, renderer=renderer)

    if prompt:
        request_text = with_resume_context(app["session"], prompt)
        result = await run_turn(
            agent=app["agent"],
            text=request_text,
            renderer=app["renderer"],
            thread_id=app["session"]["id"],
        )
        append_turn(app["session"], prompt, getattr(result, "final_text", ""), "action")
        app["session"]["turns"] = int(app["session"].get("turns") or 0) + 1
        await update_title_once(app["session"], app.get("session_model"))
        if will_compact(app["session"]):
            app["renderer"].context_compaction_started()
            try:
                await compact_if_needed(app["session"], app.get("session_model"))
            finally:
                app["renderer"].context_compaction_finished()
        app["store"].save(app["session"])
        return

    await start_repl(
        agent=app["agent"],
        plan_agent=app["plan_agent"],
        renderer=app["renderer"],
        store=app["store"],
        session=app["session"],
        model_name=app["model_name"],
        session_model=app.get("session_model"),
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
    from agent.llm import get_llm, get_model_name
    from config.loader import load_config
    from session.checkpoint import make_checkpointer
    from session.context import context_policy, mark_resume_context_pending
    from session.store import SessionStore
    from ui.renderer import Renderer

    workspace = workspace.expanduser().resolve()
    if config is None:
        config = load_config(workspace)
    if renderer is None:
        renderer = Renderer(tool_output_chars=config["tool_output_chars"])
    store = SessionStore(Path(config["session_dir"]))
    record = store.load(session, resume=resume, workspace=workspace, policy=context_policy(config))
    mark_resume_context_pending(record, resumed=bool(session or resume))
    checkpointer = make_checkpointer()
    agent = build_agent(config=config, workspace=workspace, checkpointer=checkpointer)
    plan_agent = build_plan_agent(config=config, workspace=workspace, checkpointer=checkpointer)
    session_model = get_llm(config)

    return {
        "agent": agent,
        "plan_agent": plan_agent,
        "config": config,
        "model_name": get_model_name(config),
        "renderer": renderer,
        "session": record,
        "session_model": session_model,
        "store": store,
    }
