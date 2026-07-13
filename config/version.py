"""Single source of truth for MIRA's package and display version."""

__version__ = "1.8.0"


def display_version() -> str:
    """Return the user-facing MIRA version label."""
    return f"Minimal Iterative Reasoning Agent v{__version__}"
