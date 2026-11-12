"""Diagnose duplicate paste in MIRA's Textual composer without editing production.

Run from the MIRA repository root with the same Python environment as MIRA:

    python tests/probes/mira_paste_duplication_probe.py
    python tests/probes/mira_paste_duplication_probe.py --live
    python tests/probes/mira_paste_duplication_probe.py --live --surface plain

The automatic matrix compares native app dispatch, direct widget dispatch, and
one explicit handler invocation, on an ordinary TextArea, MIRA PromptBox and
MIRA AutocompleteInput (both main and compact variants). The live mode captures
real Ctrl+V / right-click paste from a terminal. Both modes emit a structured
JSONL trace without storing pasted text or other clipboard contents.

Intentional scope: diagnostic only. It neither changes MIRA's paste behavior
nor attempts a timing-based duplicate suppression workaround.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack
import hashlib
import inspect
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable
from unittest.mock import patch

ROOT = (Path.cwd().resolve() if (Path.cwd() / "ui" / "textual" / "widgets" / "prompt_box.py").is_file()
        else Path(__file__).resolve().parents[2])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    import textual
    from textual.app import App, ComposeResult
    from textual.events import Paste
    from textual.widgets import Static, TextArea
    from ui.textual.widgets.autocomplete_input import AutocompleteInput
    from ui.textual.widgets.prompt_box import PromptBox
except ImportError as exc:
    raise SystemExit(
        f"MIRA/Textual import failed ({exc}). Run from the MIRA repo using its mira Python environment."
    ) from exc


SAMPLE = "write the file now"  # Non-command text: avoid autocomplete side effects.
SURFACES = ("plain", "prompt", "composer", "dock")
ROUTES = ("app", "widget", "direct")


def fingerprint(value: str) -> dict[str, Any]:
    """Report sizes/hashes only, never the user's pasted text."""
    return {"length": len(value), "sha12": hashlib.sha256(value.encode("utf-8")).hexdigest()[:12]}


class Trace:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.file = path.open("w", encoding="utf-8")
        self.started = time.perf_counter()
        self.case = "startup"
        self.records: list[dict[str, Any]] = []

    def emit(self, kind: str, **fields: Any) -> None:
        record = {
            "t_ms": round((time.perf_counter() - self.started) * 1000, 3),
            "case": self.case,
            "kind": kind,
            **fields,
        }
        self.records.append(record)
        self.file.write(json.dumps(record, sort_keys=True, default=str) + "\n")
        self.file.flush()

    def close(self) -> None:
        self.file.close()


class ProbeApp(App[None]):
    CSS = """
    Screen { layout: vertical; }
    #instructions { height: 3; padding: 0 1; }
    AutocompleteInput { width: 100%; height: auto; }
    #autocomplete-options { height: auto; max-height: 12; }
    #prompt { width: 100%; height: 5; }
    #plain { width: 100%; height: 5; }
    """
    BINDINGS = [("ctrl+q", "quit", "Quit probe")]

    def __init__(self, *, trace: Trace, surface: str, live: bool = False) -> None:
        super().__init__()
        self.trace = trace
        self.surface = surface
        self.live = live
        self.paste_event_ids: list[int] = []

    def compose(self) -> ComposeResult:
        if self.live:
            yield Static("Paste once into the input below; then Ctrl+Q to quit.\nFull paste contents are never logged.", id="instructions")
        if self.surface == "plain":
            yield TextArea("", show_line_numbers=False, id="plain")
        elif self.surface == "prompt":
            yield PromptBox()
        elif self.surface == "composer":
            yield AutocompleteInput(compact=False)
        elif self.surface == "dock":
            yield AutocompleteInput(compact=True)
        else:
            raise AssertionError(f"unknown surface: {self.surface}")

    def on_mount(self) -> None:
        self.query_one(TextArea).focus()

    def on_paste(self, event: Paste) -> None:
        self.paste_event_ids.append(id(event))
        self.trace.emit("app.on_paste", event_id=id(event), **fingerprint(event.text))

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        source = event.text_area
        self.trace.emit(
            "text_area.changed", widget=type(source).__name__,
            widget_id=id(source), **fingerprint(source.text)
        )


class Instrumentation:
    """Trace actual Textual/MIRA methods with temporary, reversible monkeypatches."""

    def __init__(self, trace: Trace) -> None:
        self.trace = trace
        self.stack = ExitStack()

    def __enter__(self) -> "Instrumentation":
        self._wrap_async(TextArea, "_on_paste", "text_area._on_paste")
        self._wrap_async(PromptBox, "_on_paste", "prompt_box._on_paste")
        # The base class may also receive key-based clipboard operations.
        self._wrap_if_present(TextArea, "_on_key", "text_area._on_key")
        self._wrap_if_present(PromptBox, "on_key", "prompt_box.on_key")
        # Textual 8.2.7 _on_paste actually calls _replace_via_keyboard.
        self._wrap_if_present(TextArea, "_replace_via_keyboard", "text_area._replace_via_keyboard", mutation=True)
        self._wrap_if_present(TextArea, "action_paste", "text_area.action_paste")
        self._wrap_if_present(TextArea, "insert", "text_area.insert", mutation=True)
        self._wrap_if_present(TextArea, "replace", "text_area.replace", mutation=True)
        # Textual may have explicit routing at App or Screen; trace where present.
        self._wrap_if_present(App, "_on_paste", "app._on_paste")
        return self

    def __exit__(self, *args: Any) -> None:
        self.stack.close()

    def _wrap_async(self, cls: type, name: str, label: str) -> None:
        method = getattr(cls, name)
        if not inspect.iscoroutinefunction(method):
            raise AssertionError(f"Expected async {cls.__name__}.{name}; inspect Textual version")
        trace = self.trace

        async def wrapped(instance: Any, event: Any, *args: Any, **kwargs: Any) -> Any:
            snapshot = getattr(instance, "text", "")
            trace.emit(label + ".enter", widget=type(instance).__name__, event_id=id(event),
                       event_text=fingerprint(event.text) if isinstance(event, Paste) else None,
                       before=fingerprint(snapshot))
            try:
                return await method(instance, event, *args, **kwargs)
            finally:
                trace.emit(label + ".exit", widget=type(instance).__name__, event_id=id(event),
                           after=fingerprint(getattr(instance, "text", "")))

        self.stack.enter_context(patch.object(cls, name, wrapped))

    def _wrap_if_present(self, cls: type, name: str, label: str, *, mutation: bool = False) -> None:
        method = getattr(cls, name, None)
        if method is None or not callable(method):
            self.trace.emit("hook.missing", cls=cls.__name__, method=name)
            return
        trace = self.trace
        is_async = inspect.iscoroutinefunction(method)

        def fields(args: tuple[Any, ...], instance: Any) -> dict[str, Any]:
            item: dict[str, Any] = {"widget": type(instance).__name__}
            if args and isinstance(args[0], Paste):
                item.update(event_id=id(args[0]), paste=fingerprint(args[0].text))
            elif args and hasattr(args[0], "key"):
                item["key"] = str(args[0].key)
            elif mutation and args and isinstance(args[0], str):
                item["inserted"] = fingerprint(args[0])
            return item

        if is_async:
            async def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
                before = getattr(instance, "text", "")
                trace.emit(label + ".enter", **fields(args, instance), before=fingerprint(before))
                try:
                    return await method(instance, *args, **kwargs)
                finally:
                    trace.emit(label + ".exit", **fields(args, instance),
                               after=fingerprint(getattr(instance, "text", "")))
        else:
            def wrapped(instance: Any, *args: Any, **kwargs: Any) -> Any:
                before = getattr(instance, "text", "")
                trace.emit(label + ".enter", **fields(args, instance), before=fingerprint(before))
                try:
                    return method(instance, *args, **kwargs)
                finally:
                    trace.emit(label + ".exit", **fields(args, instance),
                               after=fingerprint(getattr(instance, "text", "")))
        self.stack.enter_context(patch.object(cls, name, wrapped))


def verdict(actual: str, expected: str) -> str:
    if actual == expected:
        return "PASS / inserted once"
    if actual == expected * 2:
        return "DUPLICATED / inserted twice"
    if not actual:
        return "MISSING / not inserted"
    return f"UNEXPECTED / chars={len(actual)} (expected={len(expected)})"


async def exercise(trace: Trace, surface: str, route: str) -> dict[str, Any]:
    trace.case = f"{surface}/{route}"
    app = ProbeApp(trace=trace, surface=surface)
    async with app.run_test(size=(105, 28)) as pilot:
        target = app.query_one(TextArea)
        target.text = ""
        target.focus()
        await pilot.pause()
        trace.emit("case.ready", focused=target.has_focus)
        start = len(trace.records)  # Exclude mount/focus/initial-value messages.
        paste = Paste(SAMPLE)
        if route == "app":
            # Matches the app-level Paste used in MIRA's disabled-prompt tests.
            app.post_message(paste)
        elif route == "widget":
            target.post_message(paste)
        elif route == "direct":
            # Bypasses event dispatch: isolates just one handler invocation.
            await target._on_paste(paste)
        else:
            raise AssertionError(route)
        await pilot.pause()
        # One extra scheduler yield catches any queued Changed/bubbling events.
        await asyncio.sleep(0.04)
        await pilot.pause()
        actual = target.text
        events = trace.records[start:]
        result = {
            "surface": surface,
            "route": route,
            "verdict": verdict(actual, SAMPLE),
            "actual": fingerprint(actual),
            "paste_handler_calls": sum(x["kind"] == "text_area._on_paste.enter" for x in events),
            "mira_handler_calls": sum(x["kind"] == "prompt_box._on_paste.enter" for x in events),
            "changed_events": sum(x["kind"] == "text_area.changed" for x in events),
            "native_replacements": sum(x["kind"] == "text_area._replace_via_keyboard.enter" for x in events),
            "app_on_paste_calls": sum(x["kind"] == "app.on_paste" for x in events),
        }
        trace.emit("case.result", **result)
        return result


async def automatic(trace: Trace) -> int:
    print(f"Textual {getattr(textual, '__version__', 'unknown')} | Python {platform.python_version()}")
    print("Using one synthetic Paste per case; full text is not recorded.\n")
    results: list[dict[str, Any]] = []
    with Instrumentation(trace):
        for surface in SURFACES:
            for route in ROUTES:
                try:
                    result = await exercise(trace, surface, route)
                    results.append(result)
                    print(f"{surface:8} {route:7} {result['verdict']:27} "
                          f"base={result['paste_handler_calls']} "
                          f"mira={result['mira_handler_calls']} "
                          f"edits={result['native_replacements']} changed={result['changed_events']}")
                except Exception as exc:
                    trace.emit("case.error", type=type(exc).__name__, error=str(exc))
                    print(f"{surface:8} {route:7} ERROR: {exc}")
                    results.append({"surface": surface, "route": route, "verdict": "ERROR"})
    print("\nInterpretation:")
    print(" - plain/app duplicated: native Textual app/terminal paste routing implicated.")
    print(" - plain/app once, prompt or composer/app twice: MIRA prompt/composer implicated.")
    print(" - direct once, app twice: event dispatch / bubbling implicated, not a single handler.")
    print(" - all synthetic cases once, live Ctrl+V twice: terminal/OS/clipboard event path implicated.")
    print(" - Compare handler-entry event IDs and mutation spans in the JSONL trace.")
    if any(r["verdict"] == "ERROR" for r in results):
        return 2
    if any("DUPLICATED" in r["verdict"] for r in results):
        return 1  # Detected the regression, not a probe failure.
    return 0


def live(trace: Trace, surface: str) -> None:
    trace.case = f"LIVE/{surface}"
    trace.emit("live.begin", platform=sys.platform, textual_version=getattr(textual, "__version__", "unknown"))
    app = ProbeApp(trace=trace, surface=surface, live=True)
    with Instrumentation(trace):
        app.run()
    events = [r for r in trace.records if r["case"] == trace.case]
    event_types = ("prompt_box._on_paste.enter", "text_area._on_paste.enter",
                   "text_area.action_paste.enter", "text_area._replace_via_keyboard.enter",
                   "text_area.changed")
    counts = {kind: sum(r["kind"] == kind for r in events) for kind in event_types}
    trace.emit("live.end", **counts)
    print("Live probe paste/mutation counts:", counts)
    print(f"Inspect the event sequence and hashes in: {trace.path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Launch an interactive terminal probe for real Ctrl+V")
    parser.add_argument("--surface", choices=SURFACES, default="composer", help="Surface to show in live mode")
    parser.add_argument("--log", type=Path, default=Path("mira_paste_trace.jsonl"), help="JSONL trace path (overwritten)")
    args = parser.parse_args()
    trace = Trace(args.log.resolve())
    try:
        trace.emit("environment", python=sys.version.split()[0], os=platform.platform(),
                   textual_version=getattr(textual, "__version__", "unknown"),
                   probe="MIRA Textual duplicate-paste diagnosis")
        if args.live:
            live(trace, args.surface)
            return 0
        return asyncio.run(automatic(trace))
    finally:
        trace.close()
        print(f"Trace: {trace.path}")


if __name__ == "__main__":
    raise SystemExit(main())
