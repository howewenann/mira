"""Concise, secret-safe MCP error rendering."""

from __future__ import annotations

import re
from collections.abc import Iterator

import httpx2 as httpx

_BEARER_SECRET = re.compile(r"(?i)(\bBearer\s+)[^\s,;]+")
_NAMED_SECRET = re.compile(
    r"(?i)(access_token|refresh_token|client_secret|authorization_code|code)(\s*[=:]\s*)([^\s&,;]+)"
)
_URL_SECRET = re.compile(r"(?i)([?&](?:access_token|refresh_token|client_secret|code)=)[^&#\s]+")
_AUTH_FAILURE_FIELD = re.compile(
    r'(?i)\b(error_description|error)\s*=\s*(?:"([^"\r\n]*)"|([^,\s]+))'
)
_MAX_ERROR_DETAIL = 800


def sanitized_error(error: BaseException) -> str:
    """Render useful nested failures without leaking common credential forms."""
    nested_errors = list(_leaf_errors(error))
    substantive = [nested for nested in nested_errors if not _is_async_stream_noise(nested)]
    if substantive:
        nested_errors = substantive
    details: list[str] = []
    for nested in nested_errors:
        detail = _error_detail(nested)
        if detail and detail not in details:
            details.append(detail)
    text = "; ".join(details) or _error_detail(error)
    text = _BEARER_SECRET.sub(r"\1[redacted]", text)
    text = _NAMED_SECRET.sub(r"\1\2[redacted]", text)
    text = _URL_SECRET.sub(r"\1[redacted]", text)
    return text[:_MAX_ERROR_DETAIL]


def _leaf_errors(error: BaseException) -> Iterator[BaseException]:
    seen: set[int] = set()
    pending: list[BaseException] = [error]
    while pending:
        current = pending.pop(0)
        if id(current) in seen:
            continue
        seen.add(id(current))
        children = _error_children(current)
        if children:
            pending[0:0] = children
        else:
            yield current


def _error_children(error: BaseException) -> list[BaseException]:
    children: list[BaseException] = []
    children.extend(
        child for child in getattr(error, "exceptions", ()) if isinstance(child, BaseException)
    )
    children.extend(value for value in error.args if isinstance(value, BaseException))
    if not children and not str(error).strip():
        if isinstance(error.__cause__, BaseException):
            children.append(error.__cause__)
        elif isinstance(error.__context__, BaseException):
            children.append(error.__context__)
    return children


def _error_detail(error: BaseException) -> str:
    response = getattr(error, "response", None)
    if isinstance(response, httpx.Response):
        status = f"HTTP {response.status_code} {response.reason_phrase}".strip()
        try:
            body = " ".join(response.text.split())
        except httpx.ResponseNotRead:
            body = ""
        detail = body or _authentication_failure_detail(response)
        return f"{status}: {detail}" if detail else status
    message = " ".join(str(error).split())
    return f"{type(error).__name__}: {message}" if message else type(error).__name__


def _is_async_stream_noise(error: BaseException) -> bool:
    return type(error).__module__.startswith("anyio") and type(error).__name__ in {
        "BrokenResourceError",
        "ClosedResourceError",
        "EndOfStream",
        "WouldBlock",
    }


def _authentication_failure_detail(response: httpx.Response) -> str:
    fields: dict[str, str] = {}
    for match in _AUTH_FAILURE_FIELD.finditer(response.headers.get("www-authenticate", "")):
        fields[match.group(1).casefold()] = match.group(2) or match.group(3) or ""
    code = fields.get("error", "")
    description = fields.get("error_description", "")
    if code and description:
        return f"{code}: {description}"
    return code or description


__all__ = ["sanitized_error"]
