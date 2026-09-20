from __future__ import annotations

import asyncio
import json
import subprocess
import sys
import tempfile
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from agent.mcp.configuration import load_mcp_configuration
from agent.mcp.errors import sanitized_error
from agent.mcp.integration import MiraMCPIntegration
from agent.mcp.manager import MCPManager

TIMEOUT_SECONDS = 8
MISSING_COMMAND = "MIRA_MCP_PROBE_COMMAND_DOES_NOT_EXIST_7A9D"


def write_mcp_config(root: Path, server_spec: dict) -> None:
    directory = root / ".mira" / "mcp"
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "mcp.json").write_text(
        json.dumps({"mcpServers": {"probe": server_spec}}, indent=2),
        encoding="utf-8",
    )


def write_server(path: Path, source: str) -> None:
    path.write_text(source, encoding="utf-8")


def read_log(path: Path) -> str:
    if not path.exists():
        return "<no stderr log created>"
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    return text or "<stderr log empty>"


def raw_child_probe(server_spec: dict) -> None:
    print("\n[child process stderr]")
    command = str(server_spec["command"])
    args = [str(value) for value in server_spec.get("args", [])]
    try:
        completed = subprocess.run(
            [command, *args],
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            check=False,
        )
    except FileNotFoundError as exc:
        print(f"type={type(exc).__name__}")
        print(f"error={exc}")
        return
    except subprocess.TimeoutExpired:
        print(f"timeout after {TIMEOUT_SECONDS}s")
        return

    print(f"returncode={completed.returncode}")
    print(f"stderr={completed.stderr.strip() or '<empty>'}")
    if completed.stdout:
        print(f"stdout={completed.stdout.strip()}")


async def integration_probe(root: Path, *, capture_stderr: bool) -> None:
    state = load_mcp_configuration(root).servers["probe"]
    integration = MiraMCPIntegration(state)
    log_path = root / "fastmcp-stderr.log"

    print(
        "\n[FastMCP / transport exception"
        + (" + captured stderr]" if capture_stderr else "]")
    )
    print(f"transport.log_file before={integration._transport.log_file!r}")

    if capture_stderr:
        integration._transport.log_file = log_path
        print(f"transport.log_file probe override={log_path}")

    entered = False
    try:
        await asyncio.wait_for(integration.__aenter__(), timeout=TIMEOUT_SECONDS)
        entered = True
        print("UNEXPECTED: MCP startup succeeded")
    except BaseException as exc:
        print(f"type={type(exc).__module__}.{type(exc).__qualname__}")
        print(f"str={str(exc)!r}")
        print(f"repr={exc!r}")
        print(f"sanitized_error={sanitized_error(exc)!r}")
    finally:
        if entered:
            try:
                await integration.__aexit__(None, None, None)
            except BaseException as exc:
                print(f"cleanup_error={type(exc).__name__}: {exc}")

    if capture_stderr:
        print(f"captured_stderr={read_log(log_path)!r}")


async def manager_probe(root: Path) -> None:
    print("\n[MIRA MCP manager / generic Issues projection]")
    manager = MCPManager(root)

    async def allow_once(_state, _preview: str) -> str:
        return "allow"

    try:
        await asyncio.wait_for(
            manager.initialize(allow_once),
            timeout=TIMEOUT_SECONDS,
        )
        state = manager.servers["probe"]
        print(f"state.status={state.status!r}")
        print(f"state.error={state.error!r}")
        print(f"manager.issues={manager.issues!r}")
    except BaseException as exc:
        print(f"manager escaped exception={type(exc).__name__}: {exc}")
        print(f"manager escaped sanitized={sanitized_error(exc)!r}")
    finally:
        try:
            await manager.shutdown()
        except BaseException as exc:
            print(f"manager cleanup error={type(exc).__name__}: {exc}")


async def run_case(name: str, root: Path, server_spec: dict) -> None:
    print("\n" + "=" * 88)
    print(name)
    print("=" * 88)

    write_mcp_config(root, server_spec)
    raw_child_probe(server_spec)
    await integration_probe(root, capture_stderr=False)
    await integration_probe(root, capture_stderr=True)
    await manager_probe(root)


async def main() -> None:
    with tempfile.TemporaryDirectory(prefix="mira-mcp-probe-") as directory:
        base = Path(directory)

        import_server = base / "import_failure.py"
        write_server(
            import_server,
            """import sys
print(
    "MIRA_MCP_PROBE_IMPORT: missing dependency mira_probe_missing_dependency",
    file=sys.stderr,
    flush=True,
)
import mira_probe_missing_dependency
""",
        )
        await run_case(
            "1. Python import failure",
            base / "case-import",
            {"command": sys.executable, "args": [str(import_server)]},
        )

        raise_server = base / "startup_raise.py"
        write_server(
            raise_server,
            """import sys
print(
    "MIRA_MCP_PROBE_RAISE: startup exploded before MCP initialization",
    file=sys.stderr,
    flush=True,
)
raise RuntimeError("MIRA_MCP_PROBE_FAILURE")
""",
        )
        await run_case(
            "2. Server raises during startup",
            base / "case-raise",
            {"command": sys.executable, "args": [str(raise_server)]},
        )

        await run_case(
            "3. Invalid command / executable",
            base / "case-command",
            {"command": MISSING_COMMAND, "args": []},
        )

        protocol_server = base / "protocol_failure.py"
        write_server(
            protocol_server,
            """import sys
print(
    "MIRA_MCP_PROBE_PROTOCOL: process started but never became an MCP server",
    file=sys.stderr,
    flush=True,
)
print("THIS IS NOT AN MCP JSON-RPC MESSAGE", flush=True)
raise SystemExit(9)
""",
        )
        await run_case(
            "4. MCP protocol/startup failure",
            base / "case-protocol",
            {"command": sys.executable, "args": [str(protocol_server)]},
        )


if __name__ == "__main__":
    asyncio.run(main())
