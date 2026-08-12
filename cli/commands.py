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
    try:
        from agent.resources.tool_failures import one_shot_warning

        warning = one_shot_warning(app.get("tool_failures") or [])
        if warning:
            renderer.system_message(warning, kind="warning")
        result = await _run_one_shot(app, prompt, rubric=invocation_rubric)
        if invocation_rubric is not None and result.rubric_status != "satisfied":
            raise typer.Exit(code=3)
    finally:
        manager = app.get("mcp_manager")
        if manager is not None:
            await manager.shutdown()


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
    """Run one prompt and persist the visible transcript."""
    if app.get("agent") is None:
        from config.llm import ConfigError

        raise ConfigError(str(app.get("agent_unavailable_message") or "Main model is not configured. Run /models."))
    import typer

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

    manager = app.get("mcp_manager")
    prepared = None
    if manager is not None:
        if prompt.startswith("/mcp__"):
            await manager.discover_prompts()
        prepared = await manager.prompt_registry.resolve(prompt)
    attachments = manager.attachments_from_text(prompt) if manager is not None else []
    request_text = with_resume_context(app["session"], prompt)
    settings = (app.get("config") or {}).get("settings")
    rubric_kwargs = {}
    if rubric is not None:
        rubric_kwargs = {
            "rubric": rubric,
            "rubric_max_iterations": rubric_max_iterations(settings),
            "include_rubric_state": True,
        }
    elif rubric_enabled(settings):
        rubric_kwargs = {
            "rubric": None,
            "rubric_max_iterations": rubric_max_iterations(settings),
            "include_rubric_state": True,
        }
    recorder = SessionRecorder(app["session"], app["store"], "action")
    recorder.user_message(prompt, attachments=attachments)
    update_title(app["session"])
    recorder.save()
    renderer = RecordingRenderer(app["renderer"], recorder)
    from langchain_core.messages import HumanMessage
    from session.context import session_mcp_attachments

    all_attachments = session_mcp_attachments(app["session"])
    messages = list(prepared.messages) if prepared is not None else [
        HumanMessage(
            content=request_text,
            additional_kwargs={"mira_mcp_attachments": all_attachments} if all_attachments else {},
        )
    ]
    if prepared is not None and all_attachments:
        for index, message in enumerate(messages):
            if isinstance(message, HumanMessage):
                extra = dict(message.additional_kwargs)
                extra["mira_mcp_attachments"] = all_attachments
                messages[index] = message.model_copy(update={"additional_kwargs": extra})
                break

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
            turn_kwargs = dict(rubric_kwargs)
            if manager is not None:
                turn_kwargs["messages"] = messages
            result = await run_turn(
                agent=app["agent"],
                text=request_text,
                renderer=renderer,
                thread_id=app["session"]["id"],
                **turn_kwargs,
            )
    except asyncio.CancelledError:
        renderer.stop_active_tools("cancelled")
        raise
    except ContextOverflowError as exc:
        renderer.stop_active_tools("interrupted")
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
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        renderer.stop_active_tools("interrupted")
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
    if rubric is not None:
        renderer.system_message(
            _rubric_outcome_text(str(getattr(result, "rubric_status", "") or "")),
            kind="info",
        )
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
    """Build config, persistence, renderer, and both action/planning agents."""
    from agent.factory import build_agent, build_plan_agent
    from agent.mcp import MCPManager
    from agent.llm import active_model_issues, get_llm, get_model_name, model_unavailable_message
    from agent.resources import build_resources, configure_subagents
    from agent.resources.subagents import subagent_model_issues
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
    mcp_manager = getattr(renderer, "mcp_manager", None)
    if mcp_manager is None:
        mcp_manager = MCPManager(workspace)
        setattr(renderer, "mcp_manager", mcp_manager)
        approval = getattr(renderer, "approve_mcp_server", None)
        await mcp_manager.initialize(approval if callable(approval) else None)
    startup_progress(renderer, "discovering resources...")
    resources = build_resources(workspace, settings=config.get("settings"), config=None)
    if resources.subagent_discovery.complete and config.get("settings_valid", True):
        from config.settings import prune_subagent_settings, save_settings

        current_settings = config.get("settings") or {}
        pruned = prune_subagent_settings(
            current_settings,
            {item.name for item in resources.subagent_discovery.items},
        )
        if pruned != current_settings and save_settings(workspace, pruned):
            config["settings"] = pruned
    assignment_issues = [
        *active_model_issues(config),
        *subagent_model_issues(resources.subagent_discovery, config),
    ]
    issues = [
        *(config.get("issues") or []),
        *mcp_manager.issues,
        *mcp_manager.prompt_registry.issues,
        *resources.issues,
        *assignment_issues,
    ]
    blocking = not config.get("settings_valid", True) or bool(assignment_issues)
    metadata = ModelMetadata(
        context_tokens=int((config.get("settings") or {}).get("models", {}).get("context_limit_tokens", 32768)),
        context_source="settings.models.context_limit_tokens",
    )
    agent = None
    plan_agent = None
    if not blocking:
        action_resources = configure_subagents(resources, config)
        plan_resources = build_resources(
            workspace,
            create_examples=False,
            settings=config.get("settings"),
            enable_execute=False,
            config=config,
            subagent_discovery=resources.subagent_discovery,
        )
        startup_progress(renderer, "loading model metadata...")
        inspect_model = get_llm(config, metadata=ModelMetadata())
        metadata = await infer_model_metadata(config, model=inspect_model)
        startup_progress(renderer, "building agents...")
        agent = build_agent(
            config=config,
            workspace=workspace,
            checkpointer=checkpointer,
            metadata=metadata,
            mcp_manager=mcp_manager,
            resources=action_resources,
        )
        plan_agent = build_plan_agent(
            config=config,
            workspace=workspace,
            checkpointer=checkpointer,
            metadata=metadata,
            mcp_manager=mcp_manager,
            resources=plan_resources,
        )
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
        "tool_failures": resources.tool_failures,
        "issues": issues,
        "resource_metadata": resources.metadata,
        "project_backend": resources.project_backend,
        "agent_unavailable_message": model_unavailable_message(config) if blocking else "",
        "mcp_manager": mcp_manager,
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
