"""Probe MIRA Workflow timer behavior during complete runtime silence.

Run from the MIRA repo root:

    python tests/probes/probe_workflow_timer_silence.py

This does NOT run LangGraph, an LLM, tools, or fake Workflow events.
It checks whether MIRA's existing animation clock is wired all the way to the
live Workflow tree while nothing else is happening.
"""

from __future__ import annotations

import inspect
from collections import Counter
from typing import Any


def source(obj: Any) -> str:
    try:
        return inspect.getsource(obj)
    except (OSError, TypeError):
        return ""


def section(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


class FakeChat:
    """Records whatever tick_* methods production MiraApp calls."""

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()

    def __getattr__(self, name: str):
        if name.startswith("tick_"):
            def tick(*_args: Any, **_kwargs: Any) -> None:
                self.calls[name] += 1
            return tick
        raise AttributeError(name)


class DummyTickable:
    def tick(self) -> None:
        pass


class FakeApp:
    """Just enough surface for MiraApp._tick_animations()."""

    def __init__(self, chat_type: type[Any]) -> None:
        self.is_mounted = True
        self.mcp_manager = None
        self._mcp_spinner = 0
        self.chat_type = chat_type
        self.chat = FakeChat()
        self.dummy = DummyTickable()

    def query_one(self, query: Any, *_args: Any, **_kwargs: Any) -> Any:
        if query is self.chat_type:
            return self.chat

        # StatusBar and SubagentsPanel both only need tick() for this method.
        if getattr(query, "__name__", "") in {"StatusBar", "SubagentsPanel"}:
            return self.dummy

        raise LookupError(f"Unexpected query_one({query!r})")

    def _sync_mcp_button(self) -> None:
        pass


def check_app_tick() -> bool:
    section("1. MiraApp._tick_animations -> ChatLog")

    from ui.textual.app import MiraApp
    from ui.textual.widgets.chat_log import ChatLog

    print("Production tick calls:")
    for line in source(MiraApp._tick_animations).splitlines():
        if ".tick" in line:
            print(" ", line.strip())

    fake = FakeApp(ChatLog)

    # MIRA normally schedules this every 0.12 s.
    # 50 callbacks ~= six seconds with ZERO Workflow/LLM/runtime events.
    for _ in range(50):
        MiraApp._tick_animations(fake)

    print()
    for name, count in sorted(fake.chat.calls.items()):
        print(f"{name:<28} {count}")

    count = fake.chat.calls.get("tick_workflows", 0)
    print()

    if count == 50:
        print("PASS: Workflow ticking is connected to the global animation loop.")
        return True

    if count:
        print(f"FAIL: tick_workflows ran only {count}/50 animation callbacks.")
        return False

    print("FAIL: tick_workflows was never called.")
    print(
        "This would make Workflow clocks redraw only when some unrelated "
        "Workflow/runtime event happens."
    )
    return False


def check_chatlog_tick() -> bool:
    section("2. ChatLog.tick_workflows")

    from ui.textual.widgets.chat_log import ChatLog

    tick = getattr(ChatLog, "tick_workflows", None)

    if not callable(tick):
        print("FAIL: ChatLog.tick_workflows() does not exist.")
        print("Workflow-related ChatLog methods:")
        for name in sorted(n for n in dir(ChatLog) if "workflow" in n.lower()):
            print(" ", name)
        return False

    print("PASS: ChatLog.tick_workflows() exists.")
    print()
    print(source(tick) or "<source unavailable>")
    return True


def check_workflow_widget_tick() -> bool:
    section("3. Workflow bubble/tree periodic refresh")

    import ui.textual.widgets.workflow_bubble as module

    found_tick = False
    found_refresh = False

    for class_name, cls in vars(module).items():
        if not inspect.isclass(cls):
            continue
        if cls.__module__ != module.__name__:
            continue
        if "workflow" not in class_name.lower():
            continue

        methods = []
        for method_name in dir(cls):
            if "tick" not in method_name.lower():
                continue
            method = getattr(cls, method_name, None)
            if callable(method):
                methods.append((method_name, method))

        if not methods:
            continue

        print(f"{class_name}:")

        for method_name, method in methods:
            found_tick = True
            body = source(method)
            refresh = "refresh(" in body
            found_refresh = found_refresh or refresh

            print(f"  {method_name}()  refresh={refresh}")

            for line in body.splitlines():
                stripped = line.strip()
                if any(
                    key in stripped
                    for key in ("refresh(", "elapsed", "second", "frame")
                ):
                    print("    ", stripped)

    print()

    if not found_tick:
        print("FAIL: no Workflow widget/tree tick method found.")
        return False

    if not found_refresh:
        print("FAIL: Workflow tick exists but does not appear to call refresh().")
        print(
            "If render_label() calculates elapsed time dynamically, the Tree "
            "still has to be invalidated periodically."
        )
        return False

    print("PASS: Workflow ticking includes a UI refresh path.")
    return True


def check_live_render() -> bool:
    section("4. Workflow render_label live-time path")

    import ui.textual.widgets.workflow_bubble as module

    found = False

    for class_name, cls in vars(module).items():
        if not inspect.isclass(cls):
            continue

        render_label = getattr(cls, "render_label", None)
        if not callable(render_label):
            continue

        body = source(render_label)
        if not body:
            continue

        print(f"{class_name}.render_label():")

        interesting = [
            line.strip()
            for line in body.splitlines()
            if any(
                key in line
                for key in ("elapsed", "started_at", "monotonic", "lifecycle")
            )
        ]

        for line in interesting:
            print(" ", line)

        if interesting:
            found = True

    print()

    if found:
        print("PASS: elapsed status appears to be calculated during rendering.")
    else:
        print("CHECK: no obvious live elapsed calculation found in render_label().")

    return found


def main() -> None:
    print("MIRA Workflow silent-timer probe")
    print("No Workflow events are generated after startup.")

    try:
        app_ok = check_app_tick()
    except Exception as exc:
        app_ok = False
        print(f"\nAPP PROBE ERROR: {type(exc).__name__}: {exc}")

    try:
        chat_ok = check_chatlog_tick()
    except Exception as exc:
        chat_ok = False
        print(f"\nCHATLOG PROBE ERROR: {type(exc).__name__}: {exc}")

    try:
        widget_ok = check_workflow_widget_tick()
    except Exception as exc:
        widget_ok = False
        print(f"\nWIDGET PROBE ERROR: {type(exc).__name__}: {exc}")

    try:
        render_ok = check_live_render()
    except Exception as exc:
        render_ok = False
        print(f"\nRENDER PROBE ERROR: {type(exc).__name__}: {exc}")

    section("VERDICT")

    print(f"App animation -> Workflow tick : {'PASS' if app_ok else 'FAIL'}")
    print(f"ChatLog Workflow tick          : {'PASS' if chat_ok else 'FAIL'}")
    print(f"Workflow tree refresh tick     : {'PASS' if widget_ok else 'FAIL'}")
    print(f"Live elapsed render path       : {'PASS' if render_ok else 'CHECK'}")
    print()

    if not app_ok:
        print("Likely cause:")
        print(
            "  Workflow state is updated by runtime events, but the Workflow "
            "tree is not connected to MIRA's independent animation clock."
        )
        print()
        print("Expected chain:")
        print("  MiraApp._tick_animations()")
        print("    -> ChatLog.tick_workflows()")
        print("      -> active Workflow bubble/tree tick()")
        print("        -> Tree.refresh()")
    elif not chat_ok:
        print("Likely cause: ChatLog is missing the Workflow animation hook.")
    elif not widget_ok:
        print("Likely cause: the Workflow tick reaches the widget but does not redraw it.")
    else:
        print(
            "The basic silent-timer chain is wired. If the real LLM Workflow "
            "still freezes, next instrument whether ChatLog is ticking the SAME "
            "WorkflowTreeBubble instance that is mounted on screen."
        )


if __name__ == "__main__":
    main()
