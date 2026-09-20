"""
Expanded real-MCP live-startup probe for MIRA.

Vectors:
  1. uvx
  2. pipx run
  3. npx
  4. pnpm dlx
  5. bunx
  6. docker

Every runnable case launches a REAL MCP server through MIRA's actual
MiraMCPIntegration -> FastMCP stdio path.

The probe watches the exact FastMCP stderr sink while __aenter__() is still
pending. That distinguishes:
- genuinely live launcher/install output,
- a brief early message,
- silence until the MCP server has started,
- output that only appears after connection,
- outright launch failure.

Run:
    python tests/probes/probe_mcp_real_live_startup_expanded.py

One case:
    python tests/probes/probe_mcp_real_live_startup_expanded.py --only uvx
    python tests/probes/probe_mcp_real_live_startup_expanded.py --only pipx
    python tests/probes/probe_mcp_real_live_startup_expanded.py --only npx
    python tests/probes/probe_mcp_real_live_startup_expanded.py --only pnpm
    python tests/probes/probe_mcp_real_live_startup_expanded.py --only bunx
    python tests/probes/probe_mcp_real_live_startup_expanded.py --only docker

Notes:
- uvx/npx/pipx/bunx get probe-owned cold caches.
- pnpm gets probe-owned home/cache locations as far as its normal config allows.
- Docker cache is intentionally NOT purged.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agent.mcp.configuration import load_mcp_configuration
from agent.mcp.integration import MiraMCPIntegration


DEFAULT_TIMEOUT_SECONDS = 180.0
POLL_SECONDS = 0.04

ANSI_RE = re.compile(
    r"""
    \x1B
    (?:
        [@-Z\\-_]
        |
        \[
        [0-?]*
        [ -/]*
        [@-~]
    )
    """,
    re.VERBOSE,
)


@dataclass(frozen=True)
class Case:
    key: str
    name: str
    command: str
    args: list[str]
    note: str
    env: dict[str, str] = field(default_factory=dict)


@dataclass
class Event:
    elapsed: float
    connect_done: bool
    text: str


def tool_path(name: str) -> str | None:
    return shutil.which(name)


def shell_wrap_if_needed(command: str, args: list[str]) -> tuple[str, list[str]]:
    """
    Windows package-manager shims are commonly .cmd files. MCP's stdio launcher
    ultimately uses CreateProcess, so wrap .cmd/.bat shims in cmd.exe.
    """
    resolved = tool_path(command)
    if os.name != "nt" or resolved is None:
        return command, args

    suffix = Path(resolved).suffix.lower()
    if suffix in {".cmd", ".bat"}:
        return "cmd", ["/d", "/s", "/c", command, *args]

    return command, args


def write_mcp_config(root: Path, case: Case) -> None:
    path = root / ".mira" / "mcp" / "mcp.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    server: dict[str, object] = {
        "command": case.command,
        "args": case.args,
    }
    if case.env:
        server["env"] = case.env

    path.write_text(
        json.dumps(
            {"mcpServers": {"real_live_probe": server}},
            indent=2,
        ),
        encoding="utf-8",
    )


def display_chunk(text: str) -> str:
    text = ANSI_RE.sub("", text)
    text = text.replace("\r", "␍")
    text = text.replace("\n", "␊")
    if len(text) > 700:
        text = text[:700] + "…"
    return text


async def tail_file_live(
    path: Path,
    *,
    started_at: float,
    connect_task: asyncio.Task[object],
    stop: asyncio.Event,
    events: list[Event],
) -> None:
    offset = 0

    while not stop.is_set():
        if path.exists():
            with path.open("rb") as handle:
                handle.seek(offset)
                chunk = handle.read()
                offset = handle.tell()

            if chunk:
                elapsed = time.perf_counter() - started_at
                done = connect_task.done()
                text = chunk.decode("utf-8", errors="replace")
                events.append(Event(elapsed, done, text))
                phase = "AFTER connect completed" if done else "connect STILL pending"
                print(
                    f"[+{elapsed:7.2f}s] LIVE STDERR ({phase}): "
                    f"{display_chunk(text)}",
                    flush=True,
                )

        await asyncio.sleep(POLL_SECONDS)

    # Final drain.
    if path.exists():
        with path.open("rb") as handle:
            handle.seek(offset)
            chunk = handle.read()

        if chunk:
            elapsed = time.perf_counter() - started_at
            done = connect_task.done()
            text = chunk.decode("utf-8", errors="replace")
            events.append(Event(elapsed, done, text))
            phase = "AFTER connect completed" if done else "connect STILL pending"
            print(
                f"[+{elapsed:7.2f}s] FINAL STDERR ({phase}): "
                f"{display_chunk(text)}",
                flush=True,
            )


def docker_ready() -> tuple[bool, str]:
    if tool_path("docker") is None:
        return False, "docker executable not found"

    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )
    except Exception as exc:
        return False, f"docker preflight failed: {type(exc).__name__}: {exc}"

    if result.returncode != 0:
        detail = (result.stderr or result.stdout).strip()
        return False, f"docker daemon unavailable: {detail or 'docker info failed'}"

    return True, f"Docker server {result.stdout.strip()}"


def build_cases(base: Path) -> dict[str, Case | str]:
    allowed = base / "allowed"
    allowed.mkdir(parents=True, exist_ok=True)

    cases: dict[str, Case | str] = {}

    # ------------------------------------------------------------------ uvx
    if tool_path("uvx"):
        uv_cache = base / "uv-cache"
        command, args = shell_wrap_if_needed(
            "uvx",
            [
                "--cache-dir",
                str(uv_cache),
                "mcp-server-fetch",
            ],
        )
        cases["uvx"] = Case(
            key="uvx",
            name="uvx -> mcp-server-fetch",
            command=command,
            args=args,
            note=f"fresh uv cache: {uv_cache}",
        )
    else:
        cases["uvx"] = "SKIP: uvx executable not found"

    # ----------------------------------------------------------------- pipx
    if tool_path("pipx"):
        pipx_home = base / "pipx-home"
        pip_cache = base / "pip-cache"
        command, args = shell_wrap_if_needed(
            "pipx",
            [
                "run",
                "--spec",
                "mcp-server-fetch",
                "mcp-server-fetch",
            ],
        )
        cases["pipx"] = Case(
            key="pipx",
            name="pipx run -> mcp-server-fetch",
            command=command,
            args=args,
            note=(
                f"probe-owned PIPX_HOME={pipx_home}; "
                f"PIP_CACHE_DIR={pip_cache}"
            ),
            env={
                "PIPX_HOME": str(pipx_home),
                "PIP_CACHE_DIR": str(pip_cache),
            },
        )
    else:
        cases["pipx"] = "SKIP: pipx executable not found"

    # ------------------------------------------------------------------ npx
    if tool_path("npx"):
        npm_cache = base / "npm-cache"
        command, args = shell_wrap_if_needed(
            "npx",
            [
                "-y",
                "--cache",
                str(npm_cache),
                "@modelcontextprotocol/server-filesystem",
                str(allowed),
            ],
        )
        cases["npx"] = Case(
            key="npx",
            name="npx -> @modelcontextprotocol/server-filesystem",
            command=command,
            args=args,
            note=f"fresh npm cache: {npm_cache}",
        )
    else:
        cases["npx"] = "SKIP: npx executable not found"

    # ----------------------------------------------------------------- pnpm
    if tool_path("pnpm"):
        pnpm_home = base / "pnpm-home"
        pnpm_local = base / "pnpm-local"
        pnpm_xdg_cache = base / "pnpm-xdg-cache"
        pnpm_xdg_data = base / "pnpm-xdg-data"

        command, args = shell_wrap_if_needed(
            "pnpm",
            [
                "dlx",
                "@modelcontextprotocol/server-filesystem",
                str(allowed),
            ],
        )

        env = {"PNPM_HOME": str(pnpm_home)}
        if os.name == "nt":
            # Current pnpm puts its dlx cache under the user-local cache area on
            # Windows; isolate that area for this child process.
            env["LOCALAPPDATA"] = str(pnpm_local)
        else:
            env["XDG_CACHE_HOME"] = str(pnpm_xdg_cache)
            env["XDG_DATA_HOME"] = str(pnpm_xdg_data)

        cases["pnpm"] = Case(
            key="pnpm",
            name="pnpm dlx -> @modelcontextprotocol/server-filesystem",
            command=command,
            args=args,
            note="probe-owned pnpm home/cache locations",
            env=env,
        )
    else:
        cases["pnpm"] = "SKIP: pnpm executable not found"

    # ----------------------------------------------------------------- bunx
    if tool_path("bunx"):
        bun_cache = base / "bun-cache"
        command, args = shell_wrap_if_needed(
            "bunx",
            [
                "@modelcontextprotocol/server-filesystem",
                str(allowed),
            ],
        )
        cases["bunx"] = Case(
            key="bunx",
            name="bunx -> @modelcontextprotocol/server-filesystem",
            command=command,
            args=args,
            note=f"fresh BUN_INSTALL_CACHE_DIR: {bun_cache}",
            env={"BUN_INSTALL_CACHE_DIR": str(bun_cache)},
        )
    else:
        cases["bunx"] = "SKIP: bunx executable not found"

    # --------------------------------------------------------------- docker
    ready, detail = docker_ready()
    if ready:
        mount = f"type=bind,src={allowed.resolve()},dst=/projects/workspace"
        cases["docker"] = Case(
            key="docker",
            name="docker -> mcp/filesystem",
            command="docker",
            args=[
                "run",
                "-i",
                "--rm",
                "--mount",
                mount,
                "mcp/filesystem",
                "/projects",
            ],
            note=f"{detail}; existing Docker image cache is preserved",
        )
    else:
        cases["docker"] = f"SKIP: {detail}"

    return cases


def classify(events: list[Event]) -> tuple[str, list[Event], list[Event], float]:
    pre = [event for event in events if not event.connect_done]
    post = [event for event in events if event.connect_done]

    spread = (
        pre[-1].elapsed - pre[0].elapsed
        if len(pre) >= 2
        else 0.0
    )

    if len(pre) >= 2 and spread >= 0.20:
        status = "PROVED LIVE"
    elif len(pre) >= 1:
        status = "EARLY OUTPUT OBSERVED, BUT NOT SUSTAINED"
    elif events:
        status = "NOT LIVE IN THIS RUN: STDERR ONLY ARRIVED AT/AFTER COMPLETION"
    else:
        status = "SILENT: NO STDERR OBSERVED"

    return status, pre, post, spread


async def run_case(case: Case, root: Path, timeout: float) -> dict[str, object]:
    write_mcp_config(root, case)

    stderr_path = root / "stdio-stderr.log"
    stderr_path.write_bytes(b"")

    config = load_mcp_configuration(root)
    state = config.servers["real_live_probe"]
    integration = MiraMCPIntegration(state)

    print()
    print("=" * 110)
    print(case.name)
    print("=" * 110)
    print(f"Command: {case.command} {' '.join(case.args)}")
    print(f"Note:    {case.note}")
    if case.env:
        print(f"Extra env names: {', '.join(sorted(case.env))}")
    print(f"Original FastMCP stderr sink: {integration._transport.log_file!r}")
    print(f"Probe stderr sink:            {stderr_path}")
    print()
    print("Connection starting. 'connect STILL pending' means bytes were visible")
    print("before MIRA/FastMCP finished the MCP connection.")
    print()

    # The only behavioral override in the probe.
    integration._transport.log_file = stderr_path

    started_at = time.perf_counter()
    events: list[Event] = []
    stop = asyncio.Event()
    entered = False
    connect_error: str | None = None
    completed_at: float | None = None

    connect_task = asyncio.create_task(integration.__aenter__())
    tail_task = asyncio.create_task(
        tail_file_live(
            stderr_path,
            started_at=started_at,
            connect_task=connect_task,
            stop=stop,
            events=events,
        )
    )

    try:
        await asyncio.wait_for(connect_task, timeout=timeout)
        entered = True
        completed_at = time.perf_counter() - started_at
        print(
            f"\n[+{completed_at:7.2f}s] MCP CONNECTION / HANDSHAKE COMPLETED",
            flush=True,
        )
    except TimeoutError:
        completed_at = time.perf_counter() - started_at
        connect_error = f"TimeoutError after {timeout:.0f}s"
        print(f"\n[+{completed_at:7.2f}s] CONNECTION TIMED OUT", flush=True)
    except BaseException as exc:
        completed_at = time.perf_counter() - started_at
        connect_error = f"{type(exc).__name__}: {exc}"
        print(
            f"\n[+{completed_at:7.2f}s] CONNECTION FAILED: {connect_error}",
            flush=True,
        )
    finally:
        await asyncio.sleep(POLL_SECONDS * 3)
        stop.set()
        await tail_task

        if entered:
            try:
                await integration.__aexit__(None, None, None)
            except BaseException as exc:
                print(f"Cleanup warning: {type(exc).__name__}: {exc}")

    live_status, pre, post, spread = classify(events)
    total_bytes = sum(
        len(event.text.encode("utf-8", errors="replace"))
        for event in events
    )

    print()
    print("-" * 110)
    print(f"LIVE RESULT:        {live_status}")
    print(f"Connection:         {'SUCCESS' if entered else 'FAILED'}")
    if connect_error:
        print(f"Connection error:   {connect_error}")
    print(f"Pre-connect chunks: {len(pre)}")
    print(f"Post-connect chunks:{len(post)}")
    print(f"Observed stderr:    {total_bytes} byte(s)")
    if pre:
        print(f"First pre-connect:  +{pre[0].elapsed:.2f}s")
        print(f"Last pre-connect:   +{pre[-1].elapsed:.2f}s")
        print(f"Pre-connect spread: {spread:.2f}s")
    if completed_at is not None:
        print(f"Connect finished:   +{completed_at:.2f}s")

    return {
        "key": case.key,
        "case": case.name,
        "live_status": live_status,
        "connection": "SUCCESS" if entered else "FAILED",
        "connection_error": connect_error,
        "pre_chunks": len(pre),
        "post_chunks": len(post),
        "stderr_bytes": total_bytes,
        "spread": spread,
        "completed_at": completed_at,
    }


async def async_main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=("uvx", "pipx", "npx", "pnpm", "bunx", "docker"),
        help="Run only one launcher vector.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-vector timeout (default {DEFAULT_TIMEOUT_SECONDS:.0f}s).",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(
        prefix="mira-real-mcp-live-expanded-"
    ) as directory:
        base = Path(directory)
        cases = build_cases(base)

        selected = (
            [args.only]
            if args.only
            else ["uvx", "pipx", "npx", "pnpm", "bunx", "docker"]
        )

        print("Expanded real-MCP live startup probe")
        print(f"Temporary probe root: {base}")
        print(
            "Cold/probe-owned caches are used where practical. "
            "Docker cache is preserved."
        )

        results: list[dict[str, object]] = []
        skipped: list[tuple[str, str]] = []

        for key in selected:
            candidate = cases[key]
            if isinstance(candidate, str):
                skipped.append((key, candidate))
                print()
                print("=" * 110)
                print(f"{key}: {candidate}")
                print("=" * 110)
                continue

            case_root = base / f"workspace-{key}"
            case_root.mkdir(parents=True, exist_ok=True)
            results.append(await run_case(candidate, case_root, args.timeout))

        print()
        print("=" * 110)
        print("SUMMARY")
        print("=" * 110)

        for result in results:
            print(
                f"{result['key']:7} | "
                f"{result['live_status']:<62} | "
                f"connection={result['connection']:<7} | "
                f"pre={result['pre_chunks']:<3} | "
                f"spread={result['spread']:.2f}s"
            )

        for key, reason in skipped:
            print(f"{key:7} | {reason}")

        print()
        print("Interpretation:")
        print("- PROVED LIVE: launcher emitted multiple stderr chunks over time before MCP connected.")
        print("- EARLY OUTPUT: something appeared early, but not enough to prove sustained progress.")
        print("- NOT LIVE: stderr existed, but only at/after MCP connection.")
        print("- SILENT: launcher/server emitted no stderr at all.")
        print()
        print(
            "A NOT LIVE or SILENT result does NOT mean MIRA buffering failed. "
            "It means that launcher did not provide usable pre-handshake stderr "
            "under normal MCP stdio conditions."
        )

        return 0 if results else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main()))
