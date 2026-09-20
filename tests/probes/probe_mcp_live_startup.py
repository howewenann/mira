
'''
Probe: prove that MIRA can observe stdio MCP stderr LIVE while the MCP connection
is still starting.

Run from the MIRA repo/environment:

    python tests/probes/probe_mcp_live_startup.py

PASS means:
- stderr updates were visible incrementally,
- they arrived before the MCP handshake completed,
- no uv/npx/docker-specific parsing was involved.

This intentionally uses a plain Python stdio MCP command so the mechanism being
tested is launcher/vector agnostic.
'''


from __future__ import annotations

import asyncio
import json
import re
import sys
import tempfile
import time
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agent.mcp.configuration import load_mcp_configuration
from agent.mcp.integration import MiraMCPIntegration


PROGRESS_PREFIX = "MIRA_PROGRESS"
STARTUP_STEPS = 6
STEP_DELAY_SECONDS = 0.60
CONNECT_TIMEOUT_SECONDS = 12.0
POLL_SECONDS = 0.03


def write_mcp_config(root: Path, server_path: Path) -> None:
    path = root / ".mira" / "mcp" / "mcp.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "live_probe": {
                        "command": sys.executable,
                        "args": [str(server_path)],
                    }
                }
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def write_server(server_path: Path) -> None:
    server_path.write_text(
        f'''from __future__ import annotations
import sys
import time

for i in range(1, {STARTUP_STEPS + 1}):
    if i < {STARTUP_STEPS}:
        sys.stderr.write(
            f"\\r{PROGRESS_PREFIX} {{i}}/{STARTUP_STEPS} preparing stdio MCP environment"
        )
    else:
        sys.stderr.write(
            f"\\n{PROGRESS_PREFIX} {{i}}/{STARTUP_STEPS} starting MCP server\\n"
        )
    sys.stderr.flush()
    time.sleep({STEP_DELAY_SECONDS!r})

from fastmcp import FastMCP

server = FastMCP("MIRA live startup probe")

@server.tool
def ping(value: str) -> str:
    return f"pong: {{value}}"

if __name__ == "__main__":
    server.run(show_banner=False)
''',
        encoding="utf-8",
    )


def pop_terminal_records(buffer: str) -> tuple[list[str], str]:
    records: list[str] = []

    while True:
        match = re.search(r"[\r\n]", buffer)
        if match is None:
            break
        record = buffer[: match.start()]
        buffer = buffer[match.end() :]
        if record:
            records.append(record)

    return records, buffer


async def tail_stderr_live(
    path: Path,
    *,
    started_at: float,
    connect_task: asyncio.Task[object],
    stop: asyncio.Event,
    observations: list[tuple[float, bool, str]],
) -> None:
    offset = 0
    pending = ""

    while not stop.is_set():
        if path.exists():
            with path.open("rb") as handle:
                handle.seek(offset)
                chunk = handle.read()
                offset = handle.tell()

            if chunk:
                pending += chunk.decode("utf-8", errors="replace")
                records, pending = pop_terminal_records(pending)

                for record in records:
                    if not record.startswith(PROGRESS_PREFIX):
                        continue
                    elapsed = time.perf_counter() - started_at
                    done = connect_task.done()
                    observations.append((elapsed, done, record))
                    state = "HANDSHAKE ALREADY DONE" if done else "connect still pending"
                    print(f"[+{elapsed:5.2f}s] LIVE ({state}): {record}", flush=True)

        await asyncio.sleep(POLL_SECONDS)

    if path.exists():
        with path.open("rb") as handle:
            handle.seek(offset)
            chunk = handle.read()

        if chunk:
            pending += chunk.decode("utf-8", errors="replace")
            records, pending = pop_terminal_records(pending)
            for record in records:
                if record.startswith(PROGRESS_PREFIX):
                    elapsed = time.perf_counter() - started_at
                    done = connect_task.done()
                    observations.append((elapsed, done, record))
                    state = "HANDSHAKE ALREADY DONE" if done else "connect still pending"
                    print(f"[+{elapsed:5.2f}s] LIVE ({state}): {record}", flush=True)


async def main() -> int:
    with tempfile.TemporaryDirectory(prefix="mira-mcp-live-probe-") as directory:
        root = Path(directory)
        server_path = root / "slow_stdio_server.py"
        stderr_path = root / "stdio-stderr.log"

        write_server(server_path)
        write_mcp_config(root, server_path)
        stderr_path.touch()

        config = load_mcp_configuration(root)
        state = config.servers["live_probe"]
        integration = MiraMCPIntegration(state)

        print(f"transport.log_file before = {integration._transport.log_file!r}")
        integration._transport.log_file = stderr_path
        print(f"transport.log_file probe  = {integration._transport.log_file!r}")
        print()
        print("Starting MCP connection. Progress must appear BEFORE handshake completion.")
        print()

        started_at = time.perf_counter()
        observations: list[tuple[float, bool, str]] = []
        stop = asyncio.Event()

        connect_task = asyncio.create_task(integration.__aenter__())
        tail_task = asyncio.create_task(
            tail_stderr_live(
                stderr_path,
                started_at=started_at,
                connect_task=connect_task,
                stop=stop,
                observations=observations,
            )
        )

        entered = False
        connected_at: float | None = None

        try:
            await asyncio.wait_for(connect_task, timeout=CONNECT_TIMEOUT_SECONDS)
            entered = True
            connected_at = time.perf_counter() - started_at
            print()
            print(f"[+{connected_at:5.2f}s] MCP HANDSHAKE COMPLETED", flush=True)
        except BaseException as exc:
            print()
            print(f"MCP CONNECTION FAILED: {type(exc).__name__}: {exc}", flush=True)
        finally:
            await asyncio.sleep(POLL_SECONDS * 2)
            stop.set()
            await tail_task

            if entered:
                await integration.__aexit__(None, None, None)

        print()
        print("=" * 80)

        before_handshake = [item for item in observations if item[1] is False]

        if connected_at is None:
            print("FAIL: MCP never connected, so live-before-handshake behavior is inconclusive.")
            return 1

        if len(before_handshake) < STARTUP_STEPS - 1:
            print(
                "FAIL: startup output was not observed incrementally while the "
                "connection was pending."
            )
            print(f"Observed before handshake: {len(before_handshake)} record(s)")
            print(f"Total observed:            {len(observations)} record(s)")
            return 1

        first_at = before_handshake[0][0]
        last_at = before_handshake[-1][0]
        spread = last_at - first_at

        if spread < STEP_DELAY_SECONDS * 2:
            print(
                "FAIL: records appeared too close together; they may have been "
                "released in a batch rather than streamed live."
            )
            print(f"First pre-handshake record: +{first_at:.2f}s")
            print(f"Last pre-handshake record:  +{last_at:.2f}s")
            print(f"Spread:                     {spread:.2f}s")
            return 1

        print("PASS: stdio stderr is observable LIVE while FastMCP is still connecting.")
        print(f"Pre-handshake records: {len(before_handshake)}")
        print(f"First seen:            +{first_at:.2f}s")
        print(f"Last seen:             +{last_at:.2f}s")
        print(f"Observed spread:       {spread:.2f}s")
        print(f"Handshake completed:   +{connected_at:.2f}s")
        print()
        print(
            "Interpretation: a MIRA-owned stderr sink can drive live startup UI "
            "without knowing whether the configured command is uvx, npx, Docker, "
            "Python, Node, or another stdio launcher."
        )
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
