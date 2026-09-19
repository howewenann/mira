"""Bounded, presentation-safe capture for stdio MCP server stderr."""

from __future__ import annotations

import asyncio
import codecs
import os
import threading
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TextIO

from agent.mcp.errors import sanitized_text

_TAIL_BYTES = 64 * 1024
_LINE_CHARS = 4 * 1024
_READ_BYTES = 4096
_TRUNCATION_MARKER = " ... line truncated ... "


@dataclass(frozen=True, slots=True)
class MCPStderrSnapshot:
    """One immutable view of a server's recent sanitized stderr."""

    latest_line: str = ""
    tail: str = ""


class MCPStderrCapture:
    """Drain a child stderr pipe without retaining unbounded output."""

    def __init__(
        self,
        callback: Callable[[MCPStderrSnapshot], None],
        *,
        secret_values: Iterable[str] = (),
        tail_bytes: int = _TAIL_BYTES,
        line_chars: int = _LINE_CHARS,
    ) -> None:
        read_fd, write_fd = os.pipe()
        self._reader = os.fdopen(read_fd, "rb", buffering=0)
        self.sink: TextIO = os.fdopen(
            write_fd,
            "w",
            encoding="utf-8",
            errors="replace",
            buffering=1,
            newline="",
        )
        self._callback = callback
        self._secret_values = tuple(value for value in secret_values if value)
        self._tail_bytes = max(1, int(tail_bytes))
        self._line_chars = max(64, int(line_chars))
        self._lock = threading.Lock()
        self._lines: deque[str] = deque()
        self._tail_size = 0
        self._line = ""
        self._line_overflow = False
        self._replace_line = False
        self._escape_state = "normal"
        self._snapshot = MCPStderrSnapshot()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._closed = False

    @property
    def snapshot(self) -> MCPStderrSnapshot:
        with self._lock:
            return self._snapshot

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def started(self) -> bool:
        return self._thread is not None

    @property
    def reader_alive(self) -> bool:
        return bool(self._thread is not None and self._thread.is_alive())

    def start(self) -> None:
        """Start draining before FastMCP launches the child process."""
        if self._thread is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._thread = threading.Thread(
            target=self._drain,
            name="mcp-stderr-capture",
            daemon=True,
        )
        self._thread.start()

    async def aclose(self) -> MCPStderrSnapshot:
        """Release pipe handles and wait off-loop for the reader to finish."""
        if self._closed:
            return self.snapshot
        self._closed = True
        try:
            self.sink.close()
        except OSError:
            pass
        thread = self._thread
        if thread is not None:
            await asyncio.to_thread(thread.join, 2.0)
        try:
            self._reader.close()
        except OSError:
            pass
        if thread is not None and thread.is_alive():
            await asyncio.to_thread(thread.join, 0.2)
        snapshot = self._finish()
        self._callback(snapshot)
        return snapshot

    def close_unstarted(self) -> None:
        """Release a capture that was constructed but never entered."""
        if self._closed:
            return
        self._closed = True
        try:
            self.sink.close()
        except OSError:
            pass
        try:
            self._reader.close()
        except OSError:
            pass

    def _drain(self) -> None:
        decoder = codecs.getincrementaldecoder("utf-8")("replace")
        try:
            while chunk := self._reader.read(_READ_BYTES):
                self._consume(decoder.decode(chunk))
            self._consume(decoder.decode(b"", final=True))
        except (OSError, ValueError):
            pass
        finally:
            self._publish(self._finish())

    def _consume(self, text: str) -> None:
        changed = False
        with self._lock:
            for character in text:
                visible = self._terminal_character(character)
                if visible is None:
                    continue
                if visible == "\r":
                    self._replace_line = True
                    continue
                if visible == "\n":
                    self._commit_line_locked()
                    self._replace_line = False
                    changed = True
                    continue
                if ord(visible) < 32 and visible != "\t":
                    continue
                if self._replace_line:
                    self._line = ""
                    self._line_overflow = False
                    self._replace_line = False
                self._append_character_locked(visible)
                changed = True
            if changed:
                snapshot = self._make_snapshot_locked()
        if changed:
            self._publish(snapshot)

    def _terminal_character(self, character: str) -> str | None:
        state = self._escape_state
        if state == "normal":
            if character == "\x1b":
                self._escape_state = "escape"
                return None
            return character
        if state == "escape":
            if character == "[":
                self._escape_state = "csi"
            elif character == "]":
                self._escape_state = "osc"
            else:
                self._escape_state = "normal"
            return None
        if state == "csi":
            if "@" <= character <= "~":
                self._escape_state = "normal"
            return None
        if state == "osc":
            if character == "\x07":
                self._escape_state = "normal"
            elif character == "\x1b":
                self._escape_state = "osc_escape"
            return None
        if state == "osc_escape":
            self._escape_state = "normal" if character == "\\" else "osc"
            return None
        self._escape_state = "normal"
        return None

    def _append_character_locked(self, character: str) -> None:
        if not self._line_overflow and len(self._line) < self._line_chars:
            self._line += character
            return
        if not self._line_overflow:
            half = max(1, (self._line_chars - len(_TRUNCATION_MARKER)) // 2)
            self._line = self._line[:half] + _TRUNCATION_MARKER + self._line[-half:]
            self._line_overflow = True
        suffix = max(1, (self._line_chars - len(_TRUNCATION_MARKER)) // 2)
        marker_at = self._line.find(_TRUNCATION_MARKER)
        prefix = self._line[:marker_at]
        current_suffix = self._line[marker_at + len(_TRUNCATION_MARKER) :]
        self._line = prefix + _TRUNCATION_MARKER + (current_suffix + character)[-suffix:]

    def _commit_line_locked(self) -> None:
        line = self._sanitize(self._line)
        self._line = ""
        self._line_overflow = False
        if not line:
            return
        self._lines.append(line)
        self._tail_size += _encoded_size(line) + 1
        while self._lines and self._tail_size > self._tail_bytes:
            removed = self._lines.popleft()
            self._tail_size -= _encoded_size(removed) + 1

    def _make_snapshot_locked(self) -> MCPStderrSnapshot:
        current = self._sanitize(self._line)
        values = list(self._lines)
        if current:
            values.append(current)
        tail = "\n".join(values)
        while _encoded_size(tail) > self._tail_bytes and values:
            values.pop(0)
            tail = "\n".join(values)
        latest = current or (self._lines[-1] if self._lines else "")
        self._snapshot = MCPStderrSnapshot(latest, tail)
        return self._snapshot

    def _finish(self) -> MCPStderrSnapshot:
        with self._lock:
            if self._line:
                self._commit_line_locked()
            return self._make_snapshot_locked()

    def _sanitize(self, value: str) -> str:
        return sanitized_text(value.strip(), secret_values=self._secret_values)

    def _publish(self, snapshot: MCPStderrSnapshot) -> None:
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        loop.call_soon_threadsafe(self._callback, snapshot)


def _encoded_size(value: str) -> int:
    return len(value.encode("utf-8", errors="replace"))


__all__ = ["MCPStderrCapture", "MCPStderrSnapshot"]
