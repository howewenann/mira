"""Standard-library child process for one project-tool invocation."""

from __future__ import annotations

import asyncio
import importlib.util
import inspect
import json
import sys
import traceback
from pathlib import Path
from typing import Any


def load_file(module_name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Python file from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module


def json_result(value: Any) -> Any:
    try:
        json.dumps(value)
    except (TypeError, ValueError, OverflowError):
        return repr(value)
    return value


def run(request_path: Path, response_path: Path) -> int:
    request = json.loads(request_path.read_text(encoding="utf-8"))
    workspace = Path(request["workspace"]).resolve()
    sys.path.insert(0, str(workspace))
    try:
        load_file("mira_tool_api", Path(request["bridge_path"]).resolve())
        module = load_file("_mira_project_tool_source", Path(request["source_path"]).resolve())
        function = getattr(module, request["function_name"])
        value = function(**request.get("arguments", {}))
        if inspect.isawaitable(value):
            value = asyncio.run(value)
        response = {"ok": True, "result": json_result(value)}
        status = 0
    except BaseException as error:
        response = {
            "ok": False,
            "exception_type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(error)),
        }
        status = 1
    response_path.write_text(json.dumps(response, ensure_ascii=False), encoding="utf-8")
    return status


if __name__ == "__main__":
    raise SystemExit(run(Path(sys.argv[1]), Path(sys.argv[2])))
