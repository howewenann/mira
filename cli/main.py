"""Typer entrypoint for the MIRA command-line interface."""

from __future__ import annotations

from pathlib import Path

import typer

from cli.commands import run

app = typer.Typer(
    add_completion=False,
    no_args_is_help=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    prompt: str | None = typer.Option(None, "--prompt", "-p", help="Run one prompt and exit."),
    prompt_file: Path | None = typer.Option(
        None,
        "--file",
        "-f",
        help="Read one UTF-8 text task file and exit.",
        show_default=False,
    ),
    rubric: str | None = typer.Option(
        None,
        "--rubric",
        help="Use exact rubric text for this one-shot invocation.",
    ),
    rubric_file: Path | None = typer.Option(
        None,
        "--rubric-file",
        help="Read exact rubric text from a UTF-8 file for this one-shot invocation.",
        show_default=False,
    ),
    resume: bool = typer.Option(False, "--resume", "-r", help="Resume the most recent session."),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w", help="Workspace root.", show_default=False),
    session: str | None = typer.Option(None, "--session", "-s", help="Session id."),
    direct: bool = typer.Option(
        False,
        "--direct",
        "-d",
        help="Connect to the LLM directly, ignoring proxy env vars and disabling TLS verification.",
    ),
    trace: bool = typer.Option(
        False,
        "--trace",
        "-t",
        help="Open a trace window that shows live MIRA diagnostics.",
    ),
    acp: bool = typer.Option(False, "--acp", help="Run the stdio ACP server."),
    listen: str | None = typer.Option(
        None,
        "--listen",
        help="Experimentally serve ACP over HTTP on loopback HOST:PORT.",
    ),
) -> None:
    """Start MIRA unless Typer is dispatching to a subcommand."""
    if ctx.invoked_subcommand is None:
        if listen is not None and not acp:
            raise typer.BadParameter("--listen requires --acp", param_hint="--listen")
        if acp:
            if any((prompt, prompt_file, rubric, rubric_file, resume, session, direct, trace)):
                raise typer.BadParameter("--acp cannot be combined with interactive or one-shot options")
            if listen is not None:
                from protocols.acp.listen import validate_listen

                try:
                    listen = validate_listen(listen)
                except ValueError as exc:
                    raise typer.BadParameter(str(exc), param_hint="--listen") from exc
            try:
                from protocols.acp import run_server

                if listen is not None:
                    run_server(listen=listen)
                else:
                    run_server()
            except ModuleNotFoundError as exc:
                if listen is not None and exc.name in {
                    "acp",
                    "httpx",
                    "websockets",
                    "hypercorn",
                }:
                    typer.echo(
                        "ACP HTTP support is not installed. Install MIRA with the "
                        "'acp-http' extra.",
                        err=True,
                    )
                    raise typer.Exit(1) from exc
                if exc.name == "acp":
                    typer.echo(
                        "ACP support is not installed. Install MIRA with the 'acp' extra.",
                        err=True,
                    )
                    raise typer.Exit(1) from exc
                raise
            return
        run(
            prompt=prompt,
            prompt_file=prompt_file,
            rubric=rubric,
            rubric_file=rubric_file,
            resume=resume,
            workspace=workspace,
            session=session,
            direct=direct,
            trace=trace,
        )


if __name__ == "__main__":
    app()
