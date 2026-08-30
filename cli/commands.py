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
    rubric: str | None = None,
    rubric_file: Path | None = None,
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
                rubric=rubric,
                rubric_file=rubric_file,
                resume=resume,
                workspace=workspace,
                session=session,
                launch_options=launch_options,
                trace=trace,
            )
        )
    except ConfigError as error:
        import typer

        typer.echo(str(error), err=True)
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
    rubric: str | None = None,
    rubric_file: Path | None = None,
    trace: bool = False,
) -> None:
    """Create the app objects, then run either one-shot or TUI mode."""
    import typer

    from cli.git_guard import ensure_git_repository
    from agent.resources.project_setup import ensure_project_examples
    from config.runtime import LaunchOptions, load_effective_config

    workspace = workspace.expanduser().resolve()
    prompt, invocation_rubric = _resolve_one_shot_inputs(
        prompt,
        prompt_file,
        rubric,
        rubric_file,
        workspace,
    )
    launch_options = launch_options or LaunchOptions()
    ensure_project_examples(workspace)
    config = load_effective_config(workspace, launch_options)
    if invocation_rubric is not None:
        from config.settings import set_rubric_enabled

        config = dict(config)
        config["settings"] = set_rubric_enabled(config.get("settings") or {}, True)

    if prompt is None:
        if trace:
            from core.diagnostics.logging import get_diagnostics_logger, open_trace_window, setup_diagnostics_logging

            log_path = setup_diagnostics_logging(workspace)
            if not open_trace_window(log_path):
                get_diagnostics_logger().warning("trace window could not be opened")
        from ui.textual.app import MiraApp

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
            from tracing.stream import TraceStream

            tui.trace = TraceStream(get_diagnostics_logger(), output_chars=config["tool_output_chars"])
        await tui.run_async()
        return

    from ui.terminal.renderer import Renderer

    renderer = Renderer(tool_output_chars=config["tool_output_chars"])
    if not await ensure_git_repository(workspace, renderer):
        raise typer.Exit(code=1)

    app = await _bootstrap(workspace=workspace, session=session, resume=resume, config=config, renderer=renderer)
    try:
        from agent.resources.tool_failures import one_shot_warning

        warning = one_shot_warning(app.get("tool_failures") or [])
        if warning:
            renderer.system_message(warning, kind="warning")
        result = await _run_one_shot(app, prompt, rubric=invocation_rubric)
        if invocation_rubric is not None and result.rubric_status != "satisfied":
            raise typer.Exit(code=3)
    finally:
        try:
            application = app.get("application")
            if application is not None:
                await application.shutdown()
            else:
                manager = app.get("mcp_manager")
                if manager is not None:
                    await manager.shutdown()
        finally:
            from tracing.bootstrap import shutdown_tracing

            shutdown_tracing()


def _resolve_one_shot_inputs(
    prompt: str | None,
    prompt_file: Path | None,
    rubric: str | None,
    rubric_file: Path | None,
    workspace: Path,
) -> tuple[str | None, str | None]:
    """Validate and return exact one-shot task and optional rubric text."""
    import typer

    if prompt is not None and prompt_file is not None:
        typer.echo("Use either --prompt/-p or --file/-f, not both.", err=True)
        raise typer.Exit(code=2)
    if rubric is not None and rubric_file is not None:
        typer.echo("Use either --rubric or --rubric-file, not both.", err=True)
        raise typer.Exit(code=2)
    if (rubric is not None or rubric_file is not None) and prompt is None and prompt_file is None:
        typer.echo("--rubric and --rubric-file require --prompt/-p or --file/-f.", err=True)
        raise typer.Exit(code=2)
    if prompt is not None and not prompt.strip():
        typer.echo("--prompt/-p cannot be empty or whitespace-only.", err=True)
        raise typer.Exit(code=2)
    if rubric is not None and not rubric.strip():
        typer.echo("--rubric cannot be empty or whitespace-only.", err=True)
        raise typer.Exit(code=2)

    task = _read_text_input(prompt_file, workspace, "--file/-f") if prompt_file is not None else prompt
    criteria = (
        _read_text_input(rubric_file, workspace, "--rubric-file")
        if rubric_file is not None
        else rubric
    )
    return task, criteria


def _read_text_input(path_value: Path, workspace: Path, argument: str) -> str:
    """Read one non-empty UTF-8 text file without restricting its extension."""
    import typer

    path = path_value.expanduser()
    if not path.is_absolute():
        path = workspace / path
    path = path.resolve()

    if not path.exists():
        typer.echo(f"{argument} file does not exist: {path}", err=True)
        raise typer.Exit(code=2)
    if not path.is_file():
        typer.echo(f"{argument} path is not a file: {path}", err=True)
        raise typer.Exit(code=2)

    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        typer.echo(f"{argument} file must be valid UTF-8: {path}", err=True)
        raise typer.Exit(code=2) from error
    except OSError as error:
        typer.echo(f"Could not read {argument} file {path}: {error}", err=True)
        raise typer.Exit(code=2) from error
    if not text.strip():
        typer.echo(f"{argument} file cannot be empty or whitespace-only: {path}", err=True)
        raise typer.Exit(code=2)
    return text


async def _run_one_shot(
    app: dict[str, Any],
    prompt: str,
    *,
    rubric: str | None = None,
) -> Any:
    """Run one prompt through the same headless session contract as Textual."""
    if app.get("agent") is None:
        from config.llm import ConfigError

        raise ConfigError(str(app.get("agent_unavailable_message") or "Main model is not configured. Run /models."))
    if app.get("core_session") is None:
        raise RuntimeError("one-shot execution requires a headless MIRA session")
    return await _run_one_shot_core(app, prompt, rubric=rubric)


async def _run_one_shot_core(
    app: dict[str, Any],
    prompt: str,
    *,
    rubric: str | None = None,
) -> Any:
    """Run the normal one-shot path through the headless session facade."""
    import typer
    from langchain_core.exceptions import ContextOverflowError
    from core.diagnostics.logging import get_diagnostics_logger
    from core.diagnostics.error_report import write_error_report
    from core.api import FrontendEmitter

    manager = app.get("mcp_manager")
    prepared = None
    if manager is not None:
        if prompt.startswith("/mcp__"):
            await manager.discover_prompts()
        prepared = await manager.prompt_registry.resolve(prompt)
    attachments = manager.attachments_from_text(prompt) if manager is not None else []
    try:
        result = await app["core_session"].prompt(
            prompt,
            prepared_messages=list(prepared.messages) if prepared is not None else None,
            attachments=attachments,
            rubric_override=rubric,
        )
    except ContextOverflowError as exc:
        raise typer.Exit(code=1) from exc
    except asyncio.CancelledError:
        raise
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
        message = f"turn error: {exc}; error report: {error_path}"
        events = app["session"].setdefault("events", [])
        if events and events[-1].get("type") == "system_error":
            events[-1]["text"] = message
            app["store"].save(app["session"])
        else:
            from session.recorder import SessionRecorder

            recorder = SessionRecorder(app["session"], app["store"], "action")
            recorder.system_error(message)
            recorder.save()
        get_diagnostics_logger().exception("one-shot turn failed; error report: %s", error_path)
        FrontendEmitter(app["application"].frontend).system_message(
            message,
            kind="error",
        )
        raise
    if rubric is not None:
        from session.recorder import SessionRecorder

        outcome = _rubric_outcome_text(str(getattr(result, "rubric_status", "") or ""))
        recorder = SessionRecorder(app["session"], app["store"], "action")
        info = recorder.info(outcome)
        recorder.save()
        FrontendEmitter(app["application"].frontend).system_message(
            outcome,
            kind="info",
            created_at=str(info.get("created_at") or ""),
        )
    return result


def _rubric_outcome_text(status: str) -> str:
    """Return the final one-shot rubric outcome summary."""
    if status == "satisfied":
        return "Rubric outcome: satisfied."
    if status == "max_iterations_reached":
        return "Rubric outcome: unsatisfied after the configured iteration limit."
    label = status.replace("_", " ").strip() or "unavailable"
    return f"Rubric outcome: {label}."


async def _bootstrap(
    workspace: Path,
    session: str | None,
    resume: bool,
    config: dict[str, Any] | None = None,
    renderer: Any | None = None,
) -> dict[str, Any]:
    """Build the headless application/session and expose legacy state aliases."""
    from core.application import MiraApplication
    from core.api import FrontendEmitter
    from ui.terminal.adapter import TerminalFrontend
    from ui.terminal.renderer import Renderer

    workspace = workspace.expanduser().resolve()
    if config is None:
        from config.runtime import LaunchOptions, load_effective_config

        config = load_effective_config(workspace, LaunchOptions())
    if renderer is None:
        renderer = Renderer(tool_output_chars=config["tool_output_chars"])

    frontend = TerminalFrontend(renderer)
    progress = FrontendEmitter(frontend)
    progress.startup_progress("loading session...")
    progress.startup_progress("discovering resources...")
    application = await MiraApplication.start(
        workspace=workspace,
        frontend=frontend,
        config=config,
    )
    progress.startup_progress("loading model metadata...")
    progress.startup_progress("building agents...")
    core_session = await application.open_session(session, resume=resume)
    setattr(renderer, "mcp_manager", application.mcp_manager)

    return {
        "application": application,
        "core_session": core_session,
        "agent": application.agent,
        "plan_agent": application.plan_agent,
        "config": application.config,
        "model_name": application.model_name,
        "context_limit_tokens": application.context_limit_tokens,
        "context_limit_source": application.context_limit_source,
        "renderer": renderer,
        "session": core_session.record,
        "store": application.store,
        "workspace": application.workspace,
        "checkpointer": application.checkpointer,
        "tool_failures": application.tool_failures,
        "issues": application.issues,
        "resource_metadata": application.resource_metadata,
        "project_backend": application.project_backend,
        "agent_unavailable_message": application.agent_unavailable_message,
        "mcp_manager": application.mcp_manager,
    }


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

    from core.diagnostics.logging import get_diagnostics_logger
    from core.diagnostics.error_report import error_report_path, write_error_report

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
