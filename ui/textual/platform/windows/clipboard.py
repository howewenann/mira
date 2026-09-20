"""Native Windows clipboard writing for MIRA's Textual application."""

from __future__ import annotations

import ctypes
import time
from ctypes import wintypes
from pathlib import Path

CF_HDROP = 15
CF_UNICODETEXT = 13
GMEM_MOVEABLE = 0x0002


def get_windows_clipboard_files() -> list[Path]:
    """Return every path in the clipboard's ``CF_HDROP`` payload."""
    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        shell32 = ctypes.WinDLL("shell32", use_last_error=True)
    except (AttributeError, OSError):
        return []

    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.CloseClipboard.argtypes = []
    user32.CloseClipboard.restype = wintypes.BOOL
    user32.IsClipboardFormatAvailable.argtypes = [wintypes.UINT]
    user32.IsClipboardFormatAvailable.restype = wintypes.BOOL
    user32.GetClipboardData.argtypes = [wintypes.UINT]
    user32.GetClipboardData.restype = wintypes.HANDLE
    shell32.DragQueryFileW.argtypes = [
        wintypes.HANDLE,
        wintypes.UINT,
        wintypes.LPWSTR,
        wintypes.UINT,
    ]
    shell32.DragQueryFileW.restype = wintypes.UINT

    try:
        opened = bool(user32.OpenClipboard(None))
    except Exception:
        return []
    if not opened:
        return []

    try:
        if not user32.IsClipboardFormatAvailable(CF_HDROP):
            return []
        handle = user32.GetClipboardData(CF_HDROP)
        if not handle:
            return []
        count = shell32.DragQueryFileW(handle, 0xFFFFFFFF, None, 0)
        paths: list[Path] = []
        for index in range(count):
            length = shell32.DragQueryFileW(handle, index, None, 0)
            if not length:
                continue
            buffer = ctypes.create_unicode_buffer(length + 1)
            if shell32.DragQueryFileW(handle, index, buffer, length + 1):
                paths.append(Path(buffer.value))
        return paths
    except Exception:
        return []
    finally:
        try:
            user32.CloseClipboard()
        except Exception:
            pass


def set_windows_clipboard(text: str, *, attempts: int = 5, retry_delay: float = 0.01) -> None:
    """Write text to the Windows clipboard as ``CF_UNICODETEXT``."""
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)

    user32.OpenClipboard.argtypes = [wintypes.HWND]
    user32.OpenClipboard.restype = wintypes.BOOL
    user32.EmptyClipboard.argtypes = []
    user32.EmptyClipboard.restype = wintypes.BOOL
    user32.SetClipboardData.argtypes = [wintypes.UINT, wintypes.HANDLE]
    user32.SetClipboardData.restype = wintypes.HANDLE
    user32.CloseClipboard.argtypes = []
    user32.CloseClipboard.restype = wintypes.BOOL
    kernel32.GlobalAlloc.argtypes = [wintypes.UINT, ctypes.c_size_t]
    kernel32.GlobalAlloc.restype = wintypes.HGLOBAL
    kernel32.GlobalLock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalLock.restype = ctypes.c_void_p
    kernel32.GlobalUnlock.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalUnlock.restype = wintypes.BOOL
    kernel32.GlobalFree.argtypes = [wintypes.HGLOBAL]
    kernel32.GlobalFree.restype = wintypes.HGLOBAL

    for attempt in range(attempts):
        if user32.OpenClipboard(None):
            break
        if attempt + 1 < attempts:
            time.sleep(retry_delay)
    else:
        raise ctypes.WinError(ctypes.get_last_error())

    memory_handle: int | None = None
    owns_memory = False
    try:
        if not user32.EmptyClipboard():
            raise ctypes.WinError(ctypes.get_last_error())

        encoded = text.encode("utf-16-le") + b"\x00\x00"
        memory_handle = kernel32.GlobalAlloc(GMEM_MOVEABLE, len(encoded))
        if not memory_handle:
            raise ctypes.WinError(ctypes.get_last_error())
        owns_memory = True

        memory = kernel32.GlobalLock(memory_handle)
        if not memory:
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            ctypes.memmove(memory, encoded, len(encoded))
        finally:
            kernel32.GlobalUnlock(memory_handle)

        if not user32.SetClipboardData(CF_UNICODETEXT, memory_handle):
            raise ctypes.WinError(ctypes.get_last_error())
        owns_memory = False
    finally:
        if owns_memory and memory_handle:
            kernel32.GlobalFree(memory_handle)
        user32.CloseClipboard()
