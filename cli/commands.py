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
    """Create the app objects, then run either one-shot or TUI mode."""
    import typer

    from cli.git_guard import ensure_git_repository
    from config.loader import load_config

    workspace = workspace.expanduser().resolve()
    config = load_config(workspace)

    if prompt is None:
        from ui.app import MiraApp

        tui = MiraApp(
            workspace=workspace,
            resume=resume,
            session_id=session,
            config=config,
            bootstrap=_bootstrap,
            ensure_git_repository=ensure_git_repository,
            tool_output_chars=config["tool_output_chars"],
        )
        await tui.run_async()
        return

    from ui.renderer import Renderer

    renderer = Renderer(tool_output_chars=config["tool_output_chars"])
    if not await ensure_git_repository(workspace, renderer):
        raise typer.Exit(code=1)

    app = _bootstrap(workspace=workspace, session=session, resume=resume, config=config, renderer=renderer)
    await _run_one_shot(app, prompt)


async def _run_one_shot(app: dict[str, Any], prompt: str) -> None:
    """Run one prompt, persist the turn, and compact session context if needed."""
    from runtime.runner import run_turn
    from session.dashboard import apply_turn_usage
    from session.context import append_turn, compact_if_needed, update_title, will_compact, with_resume_context

    request_text = with_resume_context(app["session"], prompt)
    result = await run_turn(
        agent=app["agent"],
        text=request_text,
        renderer=app["renderer"],
        thread_id=app["session"]["id"],
    )
    append_turn(app["session"], prompt, getattr(result, "final_text", ""), "action")
    app["session"]["turns"] = int(app["session"].get("turns") or 0) + 1
    await update_title(app["session"], app.get("session_model"))
    if will_compact(app["session"]):
        app["renderer"].context_compaction_started()
        try:
            await compact_if_needed(app["session"], app.get("session_model"))
        finally:
            app["renderer"].context_compaction_finished()
    apply_turn_usage(
        app["session"],
        result,
        model_name=app.get("model_name", ""),
        context_limit_tokens=app.get("context_limit_tokens"),
        context_limit_source=app.get("context_limit_source", "unknown"),
    )
    app["store"].save(app["session"])


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
    from session.dashboard import context_limit_for_config, ensure_dashboard
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
    model_name = get_model_name(config)
    context_limit_tokens, context_limit_source = context_limit_for_config(config)
    ensure_dashboard(
        record,
        model_name=model_name,
        context_limit_tokens=context_limit_tokens,
        context_limit_source=context_limit_source,
    )

    return {
        "agent": agent,
        "plan_agent": plan_agent,
        "config": config,
        "model_name": model_name,
        "context_limit_tokens": context_limit_tokens,
        "context_limit_source": context_limit_source,
        "renderer": renderer,
        "session": record,
        "session_model": session_model,
        "store": store,
    }
