"""CLI command execution and application bootstrap."""

from __future__ import annotations

import asyncio
import warnings
from contextlib import suppress
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from config.runtime import LaunchOptions


def run(
    prompt: str | None,
    resume: bool,
    workspace: Path,
    session: str | None,
    direct: bool = False,
    prompt_file: Path | None = None,
    trace: bool = False,
) -> None:
    """Bridge Typer's synchronous command callback into the async app."""
    from config.llm import ConfigError
    from config.runtime import LaunchOptions

    try:
        _suppress_known_warnings()
        launch_options = LaunchOptions(llm_direct=bool(direct))
        asyncio.run(
            _run(
                prompt=prompt,
                prompt_file=prompt_file,
                resume=resume,
                workspace=workspace,
                session=session,
                launch_options=launch_options,
                trace=trace,
            )
        )
    except ConfigError as error:
        import typer

        typer.echo(f"Configuration error: {error}", err=True)
        raise typer.Exit(code=2) from error
    except KeyboardInterrupt:
        raise
    except Exception as exc:
        _write_backup_error_report(exc, workspace=workspace, session=session, prompt=prompt)
        raise


def _suppress_known_warnings() -> None:
    """Hide expected LangChain beta warnings so the CLI stays readable."""
    from langchain_core._api import LangChainBetaWarning

    warnings.filterwarnings("ignore", category=LangChainBetaWarning)


async def _run(
    prompt: str | None,
    resume: bool,
    workspace: Path,
    session: str | None,
    launch_options: LaunchOptions | None = None,
    prompt_file: Path | None = None,
    trace: bool = False,
) -> None:
    """Create the app objects, then run either one-shot or TUI mode."""
    import typer

    from cli.git_guard import ensure_git_repository
    from config.runtime import LaunchOptions, load_effective_config

    workspace = workspace.expanduser().resolve()
    prompt = _resolve_one_shot_prompt(prompt, prompt_file, workspace)
    launch_options = launch_options or LaunchOptions()
    config = load_effective_config(workspace, launch_options)

    if prompt is None:
        if trace:
            from runtime.diagnostics import get_diagnostics_logger, open_trace_window, setup_diagnostics_logging

            log_path = setup_diagnostics_logging(workspace)
            if not open_trace_window(log_path):
                get_diagnostics_logger().warning("trace window could not be opened")
        from ui.app import MiraApp

        tui = MiraApp(
            workspace=workspace,
            resume=resume,
            session_id=session,
            config=config,
            launch_options=launch_options,
            bootstrap=_bootstrap,
            ensure_git_repository=ensure_git_repository,
            tool_output_chars=config["tool_output_chars"],
        )
        if trace:
            from runtime.trace_stream import TraceStream

            tui.trace = TraceStream(get_diagnostics_logger(), output_chars=config["tool_output_chars"])
        await tui.run_async()
        return

    from ui.renderer import Renderer

    renderer = Renderer(tool_output_chars=config["tool_output_chars"])
    if not await ensure_git_repository(workspace, renderer):
        raise typer.Exit(code=1)

    app = await _bootstrap(workspace=workspace, session=session, resume=resume, config=config, renderer=renderer)
    await _run_one_shot(app, prompt)


def _resolve_one_shot_prompt(prompt: str | None, prompt_file: Path | None, workspace: Path) -> str | None:
    """Return literal prompt text or UTF-8 Markdown file contents."""
    import typer

    if prompt is not None and prompt_file is not None:
        typer.echo("Use either --prompt/-p or --file/-f, not both.", err=True)
        raise typer.Exit(code=2)
    if prompt_file is None:
        return prompt

    path = prompt_file.expanduser()
    if not path.is_absolute():
        path = workspace / path
    path = path.resolve()

    if path.suffix.lower() not in {".md", ".markdown"}:
        typer.echo("--file/-f only accepts Markdown files (.md or .markdown).", err=True)
        raise typer.Exit(code=2)
    if not path.exists():
        typer.echo(f"Prompt file does not exist: {path}", err=True)
        raise typer.Exit(code=2)
    if not path.is_file():
        typer.echo(f"Prompt file is not a file: {path}", err=True)
        raise typer.Exit(code=2)

    try:
        return path.read_text(encoding="utf-8")
    except OSError as error:
        typer.echo(f"Could not read prompt file {path}: {error}", err=True)
        raise typer.Exit(code=2) from error
    except UnicodeDecodeError as error:
        typer.echo(f"Prompt file must be UTF-8: {path}", err=True)
        raise typer.Exit(code=2) from error


async def _run_one_shot(app: dict[str, Any], prompt: str) -> None:
    """Run one prompt and persist the visible transcript."""
    from runtime.runner import run_turn
    from runtime.context_usage import context_usage_scope
    from runtime.error_report import write_error_report
    from runtime.diagnostics import get_diagnostics_logger
    from session.dashboard import apply_context_usage, apply_turn_usage
    from session.context import sync_deepagents_compaction, update_title, with_resume_context
    from session.recorder import RecordingRenderer, SessionRecorder
    from agent.context_overflow import (
        context_notice_rendered,
        mark_context_notice_rendered,
        pop_context_overflow_notice,
    )
    from langchain_core.exceptions import ContextOverflowError
    from config.settings import rubric_enabled, rubric_max_iterations

    request_text = with_resume_context(app["session"], prompt)
    settings = (app.get("config") or {}).get("settings")
    rubric_kwargs = {}
    if rubric_enabled(settings):
        rubric_kwargs = {
            "rubric": None,
            "rubric_max_iterations": rubric_max_iterations(settings),
            "include_rubric_state": True,
        }
    recorder = SessionRecorder(app["session"], app["store"], "action")
    recorder.user_message(prompt)
    update_title(app["session"])
    recorder.save()
    renderer = RecordingRenderer(app["renderer"], recorder)

    def apply_deepagents_context_usage(usage: dict[str, Any]) -> None:
        apply_context_usage(
            app["session"],
            usage.get("context_tokens", 0),
            model_name=app.get("model_name", ""),
            context_limit_tokens=app.get("context_limit_tokens"),
            context_limit_source=app.get("context_limit_source", "unknown"),
            source=str(usage.get("context_source") or "unknown"),
        )

    try:
        with context_usage_scope(apply_deepagents_context_usage):
            result = await run_turn(
                agent=app["agent"],
                text=request_text,
                renderer=renderer,
                thread_id=app["session"]["id"],
                **rubric_kwargs,
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
        report_workspace = Path(app.get("workspace") or app["session"].get("workspace") or ".")
        error_path = write_error_report(
            exc,
            workspace=report_workspace,
            source="one_shot.turn",
            session_id=str(app["session"].get("id") or ""),
            context={
                "mode": "action",
                "model": app.get("model_name", ""),
                "workspace": str(report_workspace),
            },
        )
        get_diagnostics_logger().exception("one-shot turn failed; error report: %s", error_path)
        recorder.system_error(f"turn error: {exc}; error report: {error_path}")
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
    from config.metadata import ModelMetadata, infer_model_metadata
    from config.runtime import LaunchOptions, load_effective_config
    from session.checkpoint import make_checkpointer
    from session.context import mark_resume_context_pending
    from session.dashboard import ensure_dashboard
    from session.store import SessionStore
    from ui.renderer import Renderer

    workspace = workspace.expanduser().resolve()
    if config is None:
        config = load_effective_config(workspace, LaunchOptions())
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


def _write_backup_error_report(
    exc: Exception,
    *,
    workspace: Path,
    session: str | None,
    prompt: str | None,
) -> None:
    """Best-effort top-level error report for unexpected escaping failures."""
    import typer

    if isinstance(exc, typer.Exit):
        return

    from runtime.diagnostics import get_diagnostics_logger
    from runtime.error_report import error_report_path, write_error_report

    if error_report_path(exc) is not None:
        return
    with suppress(Exception):
        resolved_workspace = workspace.expanduser().resolve()
        error_path = write_error_report(
            exc,
            workspace=resolved_workspace,
            source="cli.run",
            session_id=session,
            context={
                "workspace": str(resolved_workspace),
                "prompt_mode": "one_shot" if prompt is not None else "tui",
            },
        )
        get_diagnostics_logger().exception("top-level failure; error report: %s", error_path)
