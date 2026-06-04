"""Typer entrypoint for the MIRA command-line interface."""

from __future__ import annotations

from pathlib import Path

import typer

from cli.commands import run

app = typer.Typer(add_completion=False, no_args_is_help=False)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    prompt: str | None = typer.Option(None, "--prompt", "-p", help="Run one prompt and exit."),
    resume: bool = typer.Option(False, "--resume", "-r", help="Resume the most recent session."),
    workspace: Path = typer.Option(Path.cwd(), "--workspace", "-w", help="Workspace root."),
    session: str | None = typer.Option(None, "--session", "-s", help="Session id."),
    insecure_direct: bool = typer.Option(
        False,
        "--insecure-direct",
        help="Connect directly for LLM calls and disable TLS verification.",
    ),
) -> None:
    """Start MIRA unless Typer is dispatching to a subcommand."""
    if ctx.invoked_subcommand is None:
        run(
            prompt=prompt,
            resume=resume,
            workspace=workspace,
            session=session,
            insecure_direct=insecure_direct,
        )


if __name__ == "__main__":
    app()
