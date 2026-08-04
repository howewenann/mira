"""Small helpers for optional renderer callbacks."""

from __future__ import annotations

from inspect import Parameter, signature
from typing import Any


def call_renderer(renderer: Any, method: str, *args: Any, **kwargs: Any) -> bool:
    """Call an optional renderer method."""
    callback = getattr(renderer, method, None)
    if callback is None:
        return False
    callback(*args, **_supported_kwargs(callback, kwargs))
    return True


def _supported_kwargs(callback: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    """Keep new optional renderer metadata compatible with narrow adapters."""
    if not kwargs:
        return kwargs
    try:
        parameters = signature(callback).parameters
    except (TypeError, ValueError):
        return kwargs
    if any(parameter.kind == Parameter.VAR_KEYWORD for parameter in parameters.values()):
        return kwargs
    return {
        key: value
        for key, value in kwargs.items()
        if key in parameters
        and parameters[key].kind
        in {Parameter.POSITIONAL_OR_KEYWORD, Parameter.KEYWORD_ONLY}
    }


__all__ = ["call_renderer"]
