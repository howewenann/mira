"""
Real MCP live-startup probe for MIRA.

Exercises MIRA's real MiraMCPIntegration / FastMCP stdio path using multiple
real MCP launch vectors:

  1. uvx   -> mcp-server-fetch
  2. npx   -> @modelcontextprotocol/server-filesystem
  3. docker-> mcp/filesystem

The uv and npm cases use fresh temporary caches so a first run is genuinely
cold. Docker is NOT force-purged; if the image is already cached it may start
too quickly to demonstrate pull progress.

Run from the MIRA repo/environment:

    python tests/probes/probe_mcp_real_live_startup.py

Or run one vector:

    python tests/probes/probe_mcp_real_live_startup.py --only uvx
    python tests/probes/probe_mcp_real_live_startup.py --only npx
    python tests/probes/probe_mcp_real_live_startup.py --only docker

What proves "live":
stderr chunks are printed with timestamps while MiraMCPIntegration.__aenter__()
is STILL pending. If launcher output only becomes visible after the handshake or
failure completes, the case will say so rather than calling it live.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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
    name: str
    command: str
    args: list[str]
    note: str


@dataclass
class Event:
    elapsed: float
    connect_done: bool
    text: str


def write_mcp_config(root: Path, case: Case) -> None:
    path = root / ".mira" / "mcp" / "mcp.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "real_live_probe": {
                        "command": case.command,
                        "args": case.args,
                    }
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def display_chunk(text: str) -> str:
    """Make terminal control activity readable without hiding CR updates."""
    text = ANSI_RE.sub("", text)
    text = text.replace("\r", "␍")
    text = text.replace("\n", "␊")
    if len(text) > 500:
        text = text[:500] + "…"
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

    # Final drain so bytes written immediately around completion are classified.
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


def tool_exists(name: str) -> bool:
    return shutil.which(name) is not None


def docker_ready() -> tuple[bool, str]:
    if not tool_exists("docker"):
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

    # Real official Python MCP server. Fresh uv cache forces a cold tool env.
    if tool_exists("uvx"):
        uv_cache = base / "uv-cache"
        cases["uvx"] = Case(
            name="uvx -> mcp-server-fetch",
            command="uvx",
            args=[
                "--cache-dir",
                str(uv_cache),
                "mcp-server-fetch",
            ],
            note=f"fresh uv cache: {uv_cache}",
        )
    else:
        cases["uvx"] = "SKIP: uvx executable not found"

    # Real official Node MCP server. On Windows, use cmd /c as recommended by
    # the server's own MCP configuration examples.
    if tool_exists("npx"):
        npm_cache = base / "npm-cache"
        npx_args = [
            "-y",
            "--cache",
            str(npm_cache),
            "@modelcontextprotocol/server-filesystem",
            str(allowed),
        ]
        if os.name == "nt":
            cases["npx"] = Case(
                name="npx -> @modelcontextprotocol/server-filesystem",
                command="cmd",
                args=["/d", "/s", "/c", "npx", *npx_args],
                note=f"fresh npm cache: {npm_cache}",
            )
        else:
            cases["npx"] = Case(
                name="npx -> @modelcontextprotocol/server-filesystem",
                command="npx",
                args=npx_args,
                note=f"fresh npm cache: {npm_cache}",
            )
    else:
        cases["npx"] = "SKIP: npx executable not found"

    # Real official Docker MCP image. Do NOT delete/purge the user's Docker cache.
    ready, detail = docker_ready()
    if ready:
        mount = f"type=bind,src={allowed.resolve()},dst=/projects"
        cases["docker"] = Case(
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


async def run_case(case: Case, root: Path, timeout: float) -> dict[str, object]:
    write_mcp_config(root, case)
    stderr_path = root / "stdio-stderr.log"
    stderr_path.write_bytes(b"")

    config = load_mcp_configuration(root)
    state = config.servers["real_live_probe"]
    integration = MiraMCPIntegration(state)

    print()
    print("=" * 100)
    print(case.name)
    print("=" * 100)
    print(f"Command: {case.command} {' '.join(case.args)}")
    print(f"Note:    {case.note}")
    print(f"Original FastMCP stderr sink: {integration._transport.log_file!r}")
    print(f"Probe stderr sink:            {stderr_path}")
    print()
    print("Connection starting now. Any lines below marked 'connect STILL pending'")
    print("were observed BEFORE MIRA/FastMCP finished connecting.")
    print()

    # Only probe override: current MIRA sends this to NUL.
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

    pre = [event for event in events if not event.connect_done]
    post = [event for event in events if event.connect_done]
    total_bytes = sum(len(event.text.encode("utf-8", errors="replace")) for event in events)

    if len(pre) >= 2:
        spread = pre[-1].elapsed - pre[0].elapsed
    else:
        spread = 0.0

    # This is deliberately strict: one chunk proves early visibility, but not
    # sustained progress. Two chunks separated in time prove actual streaming.
    if len(pre) >= 2 and spread >= 0.20:
        live_status = "PROVED LIVE"
    elif len(pre) >= 1:
        live_status = "EARLY OUTPUT OBSERVED, BUT TOO BRIEF TO PROVE SUSTAINED STREAMING"
    elif events:
        live_status = "NOT LIVE IN THIS RUN: STDERR ONLY ARRIVED AT/AFTER COMPLETION"
    else:
        live_status = "INCONCLUSIVE: THIS LAUNCHER PRODUCED NO STDERR"

    print()
    print("-" * 100)
    print(f"LIVE RESULT:     {live_status}")
    print(f"Connection:      {'SUCCESS' if entered else 'FAILED'}")
    if connect_error:
        print(f"Connection err:  {connect_error}")
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
        choices=("uvx", "npx", "docker"),
        help="Run only one launcher vector.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help=f"Per-vector connection timeout (default {DEFAULT_TIMEOUT_SECONDS:.0f}s).",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="mira-real-mcp-live-") as directory:
        base = Path(directory)
        cases = build_cases(base)
        selected = [args.only] if args.only else ["uvx", "npx", "docker"]

        results: list[dict[str, object]] = []

        print("Real MCP live startup probe")
        print(f"Temporary probe root: {base}")
        print("uv/npm use fresh probe-owned caches; Docker cache is not modified.")

        for key in selected:
            candidate = cases[key]
            if isinstance(candidate, str):
                print()
                print("=" * 100)
                print(f"{key}: {candidate}")
                print("=" * 100)
                continue

            case_root = base / f"workspace-{key}"
            case_root.mkdir(parents=True, exist_ok=True)
            results.append(await run_case(candidate, case_root, args.timeout))

        print()
        print("=" * 100)
        print("SUMMARY")
        print("=" * 100)
        if not results:
            print("No runnable vectors were available.")
            return 2

        for result in results:
            print(
                f"{result['case']}: "
                f"{result['live_status']} | connection={result['connection']} | "
                f"pre_chunks={result['pre_chunks']} | spread={result['spread']:.2f}s"
            )

        print()
        print("The key evidence is the timestamped 'connect STILL pending' output above.")
        print("If those chunks arrive over time, MIRA can surface launcher-agnostic startup")
        print("progress live using the existing FastMCP stderr sink.")
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main()))
