"""CLI command execution and application bootstrap."""

from __future__ import annotations

import asyncio
import warnings
from pathlib import Path
from typing import Any


def run(
    prompt: str | None,
    resume: bool,
    workspace: Path,
    session: str | None,
    direct: bool = False,
) -> None:
    """Bridge Typer's synchronous command callback into the async app."""
    from config.llm import ConfigError

    try:
        _suppress_known_warnings()
        asyncio.run(
            _run(
                prompt=prompt,
                resume=resume,
                workspace=workspace,
                session=session,
                direct=direct,
            )
        )
    except ConfigError as error:
        import typer

        typer.echo(f"Configuration error: {error}", err=True)
        raise typer.Exit(code=2) from error


def _suppress_known_warnings() -> None:
    """Hide expected LangChain beta warnings so the CLI stays readable."""
    from langchain_core._api import LangChainBetaWarning

    warnings.filterwarnings("ignore", category=LangChainBetaWarning)


async def _run(
    prompt: str | None,
    resume: bool,
    workspace: Path,
    session: str | None,
    direct: bool = False,
) -> None:
    """Create the app objects, then run either one-shot or TUI mode."""
    import typer

    from cli.git_guard import ensure_git_repository
    from config.loader import load_config

    workspace = workspace.expanduser().resolve()
    config = load_config(workspace)
    config["llm_direct"] = bool(direct)

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
    """Run one prompt and persist the visible transcript."""
    from runtime.runner import run_turn
    from session.dashboard import apply_turn_usage
    from session.context import sync_deepagents_compaction, update_title, with_resume_context
    from session.recorder import RecordingRenderer, SessionRecorder

    request_text = with_resume_context(app["session"], prompt)
    run_kwargs = {"token_counter": app["token_counter"]} if app.get("token_counter") is not None else {}
    recorder = SessionRecorder(app["session"], app["store"], "action")
    recorder.user_message(prompt)
    update_title(app["session"])
    recorder.save()
    renderer = RecordingRenderer(app["renderer"], recorder)
    result = await run_turn(
        agent=app["agent"],
        text=request_text,
        renderer=renderer,
        thread_id=app["session"]["id"],
        **run_kwargs,
    )
    recorder.ensure_assistant(getattr(result, "final_text", ""))
    app["session"]["turns"] = int(app["session"].get("turns") or 0) + 1
    update_title(app["session"])
    if await sync_deepagents_compaction(app["session"], app["agent"], app["session"]["id"]):
        recorder.save()
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
    from agent.llm import get_model_name
    from config.loader import load_config
    from session.checkpoint import make_checkpointer
    from session.context import mark_resume_context_pending
    from session.dashboard import context_limit_for_config, ensure_dashboard, token_counter_for_config
    from session.store import SessionStore
    from ui.renderer import Renderer

    workspace = workspace.expanduser().resolve()
    if config is None:
        config = load_config(workspace)
    if renderer is None:
        renderer = Renderer(tool_output_chars=config["tool_output_chars"])
    store = SessionStore(Path(config["session_dir"]))
    record = store.load(session, resume=resume, workspace=workspace)
    mark_resume_context_pending(record, resumed=bool(session or resume))
    checkpointer = make_checkpointer()
    agent = build_agent(config=config, workspace=workspace, checkpointer=checkpointer)
    plan_agent = build_plan_agent(config=config, workspace=workspace, checkpointer=checkpointer)
    model_name = get_model_name(config)
    context_limit_tokens, context_limit_source = context_limit_for_config(config)
    token_counter = token_counter_for_config(config)
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
        "token_counter": token_counter,
        "renderer": renderer,
        "session": record,
        "store": store,
    }
