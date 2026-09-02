"""Optional Agent Client Protocol integration for MIRA."""


def run_server() -> None:
    """Launch MIRA's stdio ACP server without importing ACP during normal startup."""
    from protocols.acp.server import run_server as run

    run()


__all__ = ["run_server"]
