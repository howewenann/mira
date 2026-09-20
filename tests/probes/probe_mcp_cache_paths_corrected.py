"""
MIRA MCP install/cache-path probe — corrected pnpm version.

Goal:
Verify whether common on-demand stdio MCP launchers can keep the MCP's
downloaded/runnable artifacts under a MIRA-owned directory that users may
delete later.

For each supported vector, the probe attempts:

    cold run
    warm run
    delete MIRA-owned directory
    cold run again

The probe uses REAL MCP servers through MIRA's actual
MiraMCPIntegration -> FastMCP stdio path.

Vectors:
  uvx   -> mcp-server-fetch
  pipx  -> mcp-server-fetch
  npx   -> @modelcontextprotocol/server-filesystem
  pnpm  -> @modelcontextprotocol/server-filesystem
  bunx  -> @modelcontextprotocol/server-filesystem

Docker is documentation-only here because image storage is daemon-managed,
not a per-command removable directory.

Run from the MIRA repo/environment:

    python tests/probes/probe_mcp_cache_paths_corrected.py

Or one vector:

    python tests/probes/probe_mcp_cache_paths_corrected.py --only uvx
    python tests/probes/probe_mcp_cache_paths_corrected.py --only pipx
    python tests/probes/probe_mcp_cache_paths_corrected.py --only npx
    python tests/probes/probe_mcp_cache_paths_corrected.py --only pnpm
    python tests/probes/probe_mcp_cache_paths_corrected.py --only bunx

IMPORTANT pnpm correction:
Do NOT use:

    pnpm --cache-dir ... --store-dir ... dlx ...

Instead use the config environment variables:

    PNPM_CONFIG_CACHE_DIR=<owned>/cache
    PNPM_CONFIG_STORE_DIR=<owned>/store

and run ordinary:

    pnpm dlx ...
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
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


TIMEOUT_SECONDS = 180.0


@dataclass(frozen=True)
class Case:
    key: str
    name: str
    root: Path
    command: str
    args: list[str]
    env: dict[str, str] = field(default_factory=dict)
    note: str = ""


def which(name: str) -> str | None:
    return shutil.which(name)


def shell_wrap(command: str, args: list[str]) -> tuple[str, list[str]]:
    """
    Windows package-manager shims are often .cmd/.bat files.
    Wrap those in cmd.exe so the MCP subprocess launcher can start them.
    """
    resolved = which(command)
    if os.name != "nt" or resolved is None:
        return command, args

    if Path(resolved).suffix.lower() in {".cmd", ".bat"}:
        return "cmd", ["/d", "/s", "/c", command, *args]

    return command, args


def write_config(workspace: Path, case: Case) -> None:
    path = workspace / ".mira" / "mcp" / "mcp.json"
    path.parent.mkdir(parents=True, exist_ok=True)

    server: dict[str, object] = {
        "command": case.command,
        "args": case.args,
    }
    if case.env:
        server["env"] = case.env

    path.write_text(
        json.dumps({"mcpServers": {"probe": server}}, indent=2),
        encoding="utf-8",
    )


def dir_stats(path: Path) -> tuple[int, int]:
    if not path.exists():
        return 0, 0

    files = 0
    total = 0

    for item in path.rglob("*"):
        try:
            if item.is_file():
                files += 1
                total += item.stat().st_size
        except OSError:
            pass

    return files, total


def human_bytes(value: int) -> str:
    amount = float(value)

    for unit in ("B", "KiB", "MiB", "GiB"):
        if amount < 1024 or unit == "GiB":
            return f"{amount:.1f} {unit}"
        amount /= 1024

    return f"{value} B"


async def connect_once(
    case: Case,
    workspace: Path,
) -> tuple[bool, float, str | None]:
    write_config(workspace, case)

    config = load_mcp_configuration(workspace)
    state = config.servers["probe"]
    integration = MiraMCPIntegration(state)

    started = time.perf_counter()
    entered = False
    error: str | None = None

    try:
        await asyncio.wait_for(
            integration.__aenter__(),
            timeout=TIMEOUT_SECONDS,
        )
        entered = True
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        elapsed = time.perf_counter() - started

        if entered:
            try:
                await integration.__aexit__(None, None, None)
            except BaseException:
                pass

    return entered, elapsed, error


def build_cases(base: Path) -> dict[str, Case | str]:
    allowed = base / "allowed"
    allowed.mkdir(parents=True, exist_ok=True)

    cases: dict[str, Case | str] = {}

    # ------------------------------------------------------------------ uvx
    #
    # uvx tool environments live in the uv cache.
    # Point the cache at one MIRA-owned directory.
    if which("uvx"):
        root = base / "owned" / "uvx"

        command, args = shell_wrap(
            "uvx",
            [
                "--cache-dir",
                str(root),
                "mcp-server-fetch",
            ],
        )

        cases["uvx"] = Case(
            key="uvx",
            name="uvx -> mcp-server-fetch",
            root=root,
            command=command,
            args=args,
            note="single MIRA-owned uv cache directory",
        )
    else:
        cases["uvx"] = "SKIP: uvx not found"

    # ----------------------------------------------------------------- pipx
    #
    # PIPX_HOME causes pipx's run cache and standalone Python cache to resolve
    # under the owned root in the tested/current pipx layout.
    # Also redirect pip/uv package caches used underneath pipx.
    if which("pipx"):
        root = base / "owned" / "pipx"
        home = root / "home"
        pip_cache = root / "pip-cache"
        uv_cache = root / "uv-cache"

        command, args = shell_wrap(
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
            root=root,
            command=command,
            args=args,
            env={
                "PIPX_HOME": str(home),
                "PIP_CACHE_DIR": str(pip_cache),
                "UV_CACHE_DIR": str(uv_cache),
            },
            note=(
                f"PIPX_HOME={home}; "
                f"PIP_CACHE_DIR={pip_cache}; "
                f"UV_CACHE_DIR={uv_cache}"
            ),
        )
    else:
        cases["pipx"] = "SKIP: pipx not found"

    # ------------------------------------------------------------------ npx
    #
    # npm/npx uses the configured npm cache root for package data and the npx
    # execution cache.
    if which("npx"):
        root = base / "owned" / "npx"

        command, args = shell_wrap(
            "npx",
            [
                "-y",
                "--cache",
                str(root),
                "@modelcontextprotocol/server-filesystem",
                str(allowed),
            ],
        )

        cases["npx"] = Case(
            key="npx",
            name="npx -> @modelcontextprotocol/server-filesystem",
            root=root,
            command=command,
            args=args,
            note="single configured npm cache root",
        )
    else:
        cases["npx"] = "SKIP: npx not found"

    # ----------------------------------------------------------------- pnpm
    #
    # CORRECTED:
    # pnpm's dlx cache and its content-addressable package store are separate.
    # Put BOTH under one MIRA-owned parent via PNPM_CONFIG_* environment vars.
    #
    # This avoids relying on:
    #
    #   pnpm --cache-dir ... --store-dir ... dlx ...
    #
    # which failed in the previous probe.
    if which("pnpm"):
        root = base / "owned" / "pnpm"
        cache_dir = root / "cache"
        store_dir = root / "store"

        command, args = shell_wrap(
            "pnpm",
            [
                "dlx",
                "@modelcontextprotocol/server-filesystem",
                str(allowed),
            ],
        )

        cases["pnpm"] = Case(
            key="pnpm",
            name="pnpm dlx -> @modelcontextprotocol/server-filesystem",
            root=root,
            command=command,
            args=args,
            env={
                "PNPM_CONFIG_CACHE_DIR": str(cache_dir),
                "PNPM_CONFIG_STORE_DIR": str(store_dir),
            },
            note=(
                f"cache={cache_dir}; "
                f"store={store_dir}; "
                "both beneath one removable MIRA-owned parent"
            ),
        )
    else:
        cases["pnpm"] = "SKIP: pnpm not found"

    # ----------------------------------------------------------------- bunx
    #
    # bunx uses Bun's package cache. Redirect that cache to a MIRA-owned root.
    if which("bunx"):
        root = base / "owned" / "bunx"

        command, args = shell_wrap(
            "bunx",
            [
                "@modelcontextprotocol/server-filesystem",
                str(allowed),
            ],
        )

        cases["bunx"] = Case(
            key="bunx",
            name="bunx -> @modelcontextprotocol/server-filesystem",
            root=root,
            command=command,
            args=args,
            env={
                "BUN_INSTALL_CACHE_DIR": str(root),
            },
            note="single Bun install-cache directory",
        )
    else:
        cases["bunx"] = "SKIP: bunx not found"

    return cases


def inspect_pipx(case: Case) -> None:
    if case.key != "pipx":
        return

    env = os.environ.copy()
    env.update(case.env)

    print("pipx resolved paths:")

    for key in (
        "PIPX_HOME",
        "PIPX_VENV_CACHEDIR",
        "PIPX_STANDALONE_PYTHON_CACHEDIR",
        "UV_CACHE_DIR",
    ):
        try:
            result = subprocess.run(
                ["pipx", "environment", "--value", key],
                capture_output=True,
                text=True,
                env=env,
                timeout=15,
                check=False,
            )

            value = (result.stdout or result.stderr).strip()
            print(f"  {key}: {value}")

        except Exception as exc:
            print(
                f"  {key}: "
                f"<probe failed: {type(exc).__name__}: {exc}>"
            )


def inspect_pnpm(case: Case) -> None:
    if case.key != "pnpm":
        return

    print("pnpm configured paths:")
    print(
        f"  PNPM_CONFIG_CACHE_DIR: "
        f"{case.env.get('PNPM_CONFIG_CACHE_DIR')}"
    )
    print(
        f"  PNPM_CONFIG_STORE_DIR: "
        f"{case.env.get('PNPM_CONFIG_STORE_DIR')}"
    )


async def run_case(
    case: Case,
    workspace: Path,
) -> dict[str, object]:
    print()
    print("=" * 100)
    print(case.name)
    print("=" * 100)
    print(f"Owned root: {case.root}")
    print(f"Command:    {case.command} {' '.join(case.args)}")
    print(f"Note:       {case.note}")

    if case.env:
        print(f"Env names:   {', '.join(sorted(case.env))}")

    inspect_pipx(case)
    inspect_pnpm(case)

    if case.root.exists():
        shutil.rmtree(case.root)

    # ------------------------------------------------------------- cold run
    print("\n1) COLD RUN")

    ok1, t1, err1 = await connect_once(case, workspace)
    files1, bytes1 = dir_stats(case.root)

    print(
        f"   connection={'SUCCESS' if ok1 else 'FAILED'} "
        f"time={t1:.2f}s"
    )

    if err1:
        print(f"   error={err1}")

    print(
        f"   owned root exists={case.root.exists()} "
        f"files={files1} "
        f"size={human_bytes(bytes1)}"
    )

    # ------------------------------------------------------------- warm run
    print("\n2) WARM RUN")

    ok2, t2, err2 = await connect_once(case, workspace)
    files2, bytes2 = dir_stats(case.root)

    print(
        f"   connection={'SUCCESS' if ok2 else 'FAILED'} "
        f"time={t2:.2f}s"
    )

    if err2:
        print(f"   error={err2}")

    print(
        f"   owned root exists={case.root.exists()} "
        f"files={files2} "
        f"size={human_bytes(bytes2)}"
    )

    # -------------------------------------------------------------- delete
    print("\n3) DELETE OWNED ROOT")

    if case.root.exists():
        shutil.rmtree(case.root)

    print(f"   exists after delete={case.root.exists()}")

    # -------------------------------------------------------- cold after rm
    print("\n4) COLD RUN AFTER DELETE")

    ok3, t3, err3 = await connect_once(case, workspace)
    files3, bytes3 = dir_stats(case.root)

    print(
        f"   connection={'SUCCESS' if ok3 else 'FAILED'} "
        f"time={t3:.2f}s"
    )

    if err3:
        print(f"   error={err3}")

    print(
        f"   owned root recreated={case.root.exists()} "
        f"files={files3} "
        f"size={human_bytes(bytes3)}"
    )

    # -------------------------------------------------------------- result
    if (
        ok1
        and ok2
        and ok3
        and case.root.exists()
        and files1 > 0
        and files3 > 0
    ):
        result = "PASS: removable MIRA-owned storage behavior observed"
    else:
        result = "INCONCLUSIVE/FAIL: inspect output"

    print(f"\nRESULT: {result}")

    return {
        "key": case.key,
        "result": result,
        "cold": t1,
        "warm": t2,
        "recold": t3,
        "files_first": files1,
        "files_after_delete": files3,
        "cold_ok": ok1,
        "warm_ok": ok2,
        "recold_ok": ok3,
    }


async def main() -> int:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--only",
        choices=("uvx", "pipx", "npx", "pnpm", "bunx"),
    )

    args = parser.parse_args()

    with tempfile.TemporaryDirectory(
        prefix="mira-mcp-cache-paths-corrected-"
    ) as directory:
        base = Path(directory)
        cases = build_cases(base)

        selected = (
            [args.only]
            if args.only
            else ["uvx", "pipx", "npx", "pnpm", "bunx"]
        )

        print("MIRA MCP install/cache-path probe — corrected pnpm")
        print(f"Probe root: {base}")
        print(
            "All intended package/runtime storage is placed "
            "under the temporary probe root."
        )
        print(
            "Docker is intentionally excluded: "
            "Docker image storage is daemon-managed."
        )

        results: list[dict[str, object]] = []

        for key in selected:
            candidate = cases[key]

            if isinstance(candidate, str):
                print()
                print(f"{key}: {candidate}")
                continue

            workspace = base / f"workspace-{key}"
            workspace.mkdir(parents=True, exist_ok=True)

            results.append(
                await run_case(
                    candidate,
                    workspace,
                )
            )

        print()
        print("=" * 100)
        print("SUMMARY")
        print("=" * 100)

        for item in results:
            print(
                f"{item['key']:5} | "
                f"{item['result']} | "
                f"cold={item['cold']:.2f}s "
                f"warm={item['warm']:.2f}s "
                f"after-delete={item['recold']:.2f}s"
            )

        print()
        print("Docker documentation rule:")
        print(
            "  There is no per-command --cache-dir equivalent "
            "for image storage."
        )
        print(
            "  Cleanup is by image identity, "
            "e.g. `docker image rm <image>`."
        )

        return 0 if results else 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
