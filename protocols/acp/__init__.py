"""Optional Agent Client Protocol integration for MIRA."""


def run_server(*, listen: str | None = None) -> None:
    """Launch stdio ACP, or the optional HTTP transport when requested."""
    if listen is not None:
        from protocols.acp.http.server import run_http_server as run

        run(listen)
        return

    from protocols.acp.stdio.server import run_server as run

    run()


__all__ = ["run_server"]
