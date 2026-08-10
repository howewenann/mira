"""Environment interpolation shared by MIRA configuration surfaces."""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

_REFERENCE = re.compile(r"\$\{([^}]*)\}")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


class EnvironmentInterpolationError(ValueError):
    """Raised when an environment reference is unsupported or unresolved."""


def resolve_environment(value: Any, *, environ: Mapping[str, str] | None = None) -> Any:
    """Resolve ``${NAME}`` recursively in configuration values."""
    source = os.environ if environ is None else environ
    if isinstance(value, str):
        if "$${" in value:
            raise EnvironmentInterpolationError("unsupported environment escape; use ${NAME}")
        if "${env:" in value:
            raise EnvironmentInterpolationError("unsupported environment reference; use ${NAME}")
        if "${" in _REFERENCE.sub("", value):
            raise EnvironmentInterpolationError("malformed environment reference; use ${NAME}")

        def replace(match: re.Match[str]) -> str:
            name = match.group(1)
            if _NAME.fullmatch(name) is None:
                raise EnvironmentInterpolationError(
                    f"invalid environment variable name {name!r}; use ${{NAME}}"
                )
            if name not in source:
                raise EnvironmentInterpolationError(
                    f"environment variable {name} is not set; define it before starting MIRA "
                    "or in the workspace .env"
                )
            return source[name]

        return _REFERENCE.sub(replace, value)
    if isinstance(value, list):
        return [resolve_environment(item, environ=source) for item in value]
    if isinstance(value, dict):
        return {key: resolve_environment(item, environ=source) for key, item in value.items()}
    return value


__all__ = ["EnvironmentInterpolationError", "resolve_environment"]
