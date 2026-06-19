"""CLI command execution and application bootstrap."""

from __future__ import annotations

import asyncio
import warnings
from contextlib import suppress
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

    app = await _bootstrap(workspace=workspace, session=session, resume=resume, config=config, renderer=renderer)
    await _run_one_shot(app, prompt)


async def _run_one_shot(app: dict[str, Any], prompt: str) -> None:
    """Run one prompt and persist the visible transcript."""
    from runtime.runner import run_turn
    from session.dashboard import apply_turn_usage
    from session.context import sync_deepagents_compaction, update_title, with_resume_context
    from session.recorder import RecordingRenderer, SessionRecorder
    from agent.context_overflow import (
        context_notice_rendered,
        mark_context_notice_rendered,
        pop_context_overflow_notice,
    )
    from langchain_core.exceptions import ContextOverflowError

    request_text = with_resume_context(app["session"], prompt)
    recorder = SessionRecorder(app["session"], app["store"], "action")
    recorder.user_message(prompt)
    update_title(app["session"])
    recorder.save()
    renderer = RecordingRenderer(app["renderer"], recorder)
    try:
        result = await run_turn(
            agent=app["agent"],
            text=request_text,
            renderer=renderer,
            thread_id=app["session"]["id"],
        )
    except ContextOverflowError as exc:
        with suppress(Exception):
            await sync_deepagents_compaction(app["session"], app["agent"], app["session"]["id"])
        notice = pop_context_overflow_notice(exc)
        if notice and not context_notice_rendered(exc):
            system_message = getattr(renderer, "system_message", None)
            if callable(system_message):
                system_message(notice, kind="info")
            else:
                recorder.info(notice)
            mark_context_notice_rendered(exc)
        app["store"].save(app["session"])
        return
    except Exception as exc:
        recorder.system_error(f"turn error: {exc}")
        raise
    recorder.ensure_assistant(getattr(result, "final_text", ""))
    app["session"]["turns"] = int(app["session"].get("turns") or 0) + 1
    update_title(app["session"])
    with suppress(Exception):
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


async def _bootstrap(
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
    from config.metadata import ModelMetadata, infer_model_metadata
    from session.checkpoint import make_checkpointer
    from session.context import mark_resume_context_pending
    from session.dashboard import ensure_dashboard
    from session.store import SessionStore
    from ui.renderer import Renderer

    workspace = workspace.expanduser().resolve()
    if config is None:
        config = load_config(workspace)
    if renderer is None:
        renderer = Renderer(tool_output_chars=config["tool_output_chars"])
    startup_progress(renderer, "loading session...")
    store = SessionStore(Path(config["session_dir"]))
    record = store.load(session, resume=resume, workspace=workspace)
    mark_resume_context_pending(record, resumed=bool(session or resume))
    checkpointer = make_checkpointer()
    startup_progress(renderer, "loading model metadata...")
    inspect_model = get_llm(config, metadata=ModelMetadata())
    metadata = await infer_model_metadata(config, model=inspect_model)
    config["llm_inferred_context_tokens"] = metadata.context_tokens
    config["llm_context_source"] = metadata.context_source
    startup_progress(renderer, "building agents...")
    agent = build_agent(config=config, workspace=workspace, checkpointer=checkpointer, metadata=metadata)
    plan_agent = build_plan_agent(config=config, workspace=workspace, checkpointer=checkpointer, metadata=metadata)
    model_name = get_model_name(config)
    context_limit_tokens = metadata.context_tokens
    context_limit_source = metadata.context_source
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
        "store": store,
        "workspace": workspace,
        "checkpointer": checkpointer,
    }


def startup_progress(renderer: Any, state: str) -> None:
    """Notify renderers that expose startup progress."""
    callback = getattr(renderer, "startup_progress", None)
    if callable(callback):
        callback(state)
