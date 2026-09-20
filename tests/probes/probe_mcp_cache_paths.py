"""
MIRA MCP install/cache-path probe.

Goal:
Verify whether common on-demand stdio MCP launchers can keep the MCP's
downloaded/runnable artifacts under a MIRA-owned directory that users may
delete later.

For each supported vector, the probe attempts:

    cold run
    warm run
    delete MIRA-owned directory
    cold run again

The probe uses REAL MCP servers through MIRA's actual MiraMCPIntegration path.

Vectors:
  uvx   -> mcp-server-fetch
  npx   -> @modelcontextprotocol/server-filesystem
  pnpm  -> @modelcontextprotocol/server-filesystem
  bunx  -> @modelcontextprotocol/server-filesystem
  pipx  -> mcp-server-fetch (inspection + run; pipx has split cache semantics)
  docker-> documentation-only note (daemon-managed image storage)

Run from the MIRA repo/environment:

    python tests/probes/probe_mcp_cache_paths.py

Or one vector:

    python tests/probes/probe_mcp_cache_paths.py --only uvx
    python tests/probes/probe_mcp_cache_paths.py --only npx
    python tests/probes/probe_mcp_cache_paths.py --only pnpm
    python tests/probes/probe_mcp_cache_paths.py --only pipx
    python tests/probes/probe_mcp_cache_paths.py --only bunx
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


async def connect_once(case: Case, workspace: Path) -> tuple[bool, float, str | None]:
    write_config(workspace, case)
    config = load_mcp_configuration(workspace)
    state = config.servers["probe"]
    integration = MiraMCPIntegration(state)

    started = time.perf_counter()
    entered = False
    error: str | None = None
    try:
        await asyncio.wait_for(integration.__aenter__(), timeout=TIMEOUT_SECONDS)
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

    # uvx: official uv docs say tool-run environments live inside the uv cache.
    if which("uvx"):
        root = base / "owned" / "uvx"
        command, args = shell_wrap(
            "uvx",
            ["--cache-dir", str(root), "mcp-server-fetch"],
        )
        cases["uvx"] = Case(
            "uvx",
            "uvx -> mcp-server-fetch",
            root,
            command,
            args,
            note="single MIRA-owned uv cache directory",
        )
    else:
        cases["uvx"] = "SKIP: uvx not found"

    # npx/npm: npm's configured cache contains both package cache and npx cache.
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
            "npx",
            "npx -> @modelcontextprotocol/server-filesystem",
            root,
            command,
            args,
            note="single configured npm cache root",
        )
    else:
        cases["npx"] = "SKIP: npx not found"

    # pnpm: cacheDir holds dlx metadata/env cache, storeDir holds package content.
    # Put BOTH beneath one MIRA-owned parent so deleting the parent removes both.
    if which("pnpm"):
        root = base / "owned" / "pnpm"
        cache_dir = root / "cache"
        store_dir = root / "store"
        command, args = shell_wrap(
            "pnpm",
            [
                "--cache-dir",
                str(cache_dir),
                "--store-dir",
                str(store_dir),
                "dlx",
                "@modelcontextprotocol/server-filesystem",
                str(allowed),
            ],
        )
        cases["pnpm"] = Case(
            "pnpm",
            "pnpm dlx -> @modelcontextprotocol/server-filesystem",
            root,
            command,
            args,
            note=f"cache={cache_dir}; store={store_dir}",
        )
    else:
        cases["pnpm"] = "SKIP: pnpm not found"

    # bunx: official Bun docs expose BUN_INSTALL_CACHE_DIR for the global
    # package cache used by bunx.
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
            "bunx",
            "bunx -> @modelcontextprotocol/server-filesystem",
            root,
            command,
            args,
            env={"BUN_INSTALL_CACHE_DIR": str(root)},
            note="single Bun install-cache directory",
        )
    else:
        cases["bunx"] = "SKIP: bunx not found"

    # pipx: deliberately *not* claiming one clean --cache-dir. PIPX_HOME is
    # set, but pipx run's PIPX_VENV_CACHEDIR is derived separately. We inspect
    # that resolved value in the probe before deciding documentation.
    if which("pipx"):
        root = base / "owned" / "pipx"
        home = root / "home"
        pip_cache = root / "pip-cache"
        uv_cache = root / "uv-cache"
        command, args = shell_wrap(
            "pipx",
            ["run", "--spec", "mcp-server-fetch", "mcp-server-fetch"],
        )
        cases["pipx"] = Case(
            "pipx",
            "pipx run -> mcp-server-fetch",
            root,
            command,
            args,
            env={
                "PIPX_HOME": str(home),
                "PIP_CACHE_DIR": str(pip_cache),
                "UV_CACHE_DIR": str(uv_cache),
            },
            note="inspection case: pipx run cache location is derived",
        )
    else:
        cases["pipx"] = "SKIP: pipx not found"

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
            print(f"  {key}: <probe failed: {type(exc).__name__}: {exc}>")


async def run_case(case: Case, workspace: Path) -> dict[str, object]:
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

    if case.root.exists():
        shutil.rmtree(case.root)

    print("\n1) COLD RUN")
    ok1, t1, err1 = await connect_once(case, workspace)
    files1, bytes1 = dir_stats(case.root)
    print(f"   connection={'SUCCESS' if ok1 else 'FAILED'} time={t1:.2f}s")
    if err1:
        print(f"   error={err1}")
    print(
        f"   owned root exists={case.root.exists()} "
        f"files={files1} size={human_bytes(bytes1)}"
    )

    print("\n2) WARM RUN")
    ok2, t2, err2 = await connect_once(case, workspace)
    files2, bytes2 = dir_stats(case.root)
    print(f"   connection={'SUCCESS' if ok2 else 'FAILED'} time={t2:.2f}s")
    if err2:
        print(f"   error={err2}")
    print(
        f"   owned root exists={case.root.exists()} "
        f"files={files2} size={human_bytes(bytes2)}"
    )

    print("\n3) DELETE OWNED ROOT")
    if case.root.exists():
        shutil.rmtree(case.root)
    print(f"   exists after delete={case.root.exists()}")

    print("\n4) COLD RUN AFTER DELETE")
    ok3, t3, err3 = await connect_once(case, workspace)
    files3, bytes3 = dir_stats(case.root)
    print(f"   connection={'SUCCESS' if ok3 else 'FAILED'} time={t3:.2f}s")
    if err3:
        print(f"   error={err3}")
    print(
        f"   owned root recreated={case.root.exists()} "
        f"files={files3} size={human_bytes(bytes3)}"
    )

    if ok1 and ok2 and ok3 and case.root.exists() and files1 and files3:
        result = "PASS: removable MIRA-owned storage behavior observed"
    elif case.key == "pipx":
        result = "INSPECT: use resolved pipx paths above to decide docs"
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
    }


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--only",
        choices=("uvx", "pipx", "npx", "pnpm", "bunx"),
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(
        prefix="mira-mcp-cache-paths-"
    ) as directory:
        base = Path(directory)
        cases = build_cases(base)
        selected = (
            [args.only]
            if args.only
            else ["uvx", "pipx", "npx", "pnpm", "bunx"]
        )

        print("MIRA MCP install/cache-path probe")
        print(f"Probe root: {base}")
        print("All intended package storage is under the temporary probe root.")
        print("Docker is intentionally excluded: Docker image storage is daemon-managed.")

        results: list[dict[str, object]] = []

        for key in selected:
            candidate = cases[key]
            if isinstance(candidate, str):
                print(f"\n{key}: {candidate}")
                continue

            workspace = base / f"workspace-{key}"
            workspace.mkdir(parents=True, exist_ok=True)
            results.append(await run_case(candidate, workspace))

        print()
        print("=" * 100)
        print("SUMMARY")
        print("=" * 100)
        for item in results:
            print(
                f"{item['key']:5} | {item['result']} | "
                f"cold={item['cold']:.2f}s warm={item['warm']:.2f}s "
                f"after-delete={item['recold']:.2f}s"
            )

        print()
        print("Docker documentation rule:")
        print("  There is no per-command --cache-dir equivalent for image storage.")
        print("  Cleanup is by image identity, e.g. `docker image rm <image>`.")
        return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
