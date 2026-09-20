"""Native Windows file selection for MIRA's Textual interface."""

from __future__ import annotations

from pathlib import Path


def choose_local_files(initial_directory: Path) -> list[Path]:
    """Open the stdlib native multi-file picker and return selected paths."""
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askopenfilenames(
            title="Add files to MIRA",
            initialdir=str(initial_directory),
        )
        return [Path(path) for path in selected]
    finally:
        root.destroy()
