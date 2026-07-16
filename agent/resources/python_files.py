"""Import helpers for trusted Python resource files."""

from __future__ import annotations

import importlib.util
import sys
from hashlib import sha1
from pathlib import Path
from typing import Any


def import_python_file(path: Path, prefix: str) -> Any:
    module_id = sha1(str(path.resolve()).encode("utf-8")).hexdigest()[:12]
    module_name = f"{prefix}_{module_id}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import Python resource from {path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        sys.modules.pop(module_name, None)
        raise
    return module
